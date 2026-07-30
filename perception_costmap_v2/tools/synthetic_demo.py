#!/usr/bin/env python3
"""
synthetic_demo.py -- end-to-end pipeline demo WITHOUT CARLA.

This machine has no GPU and cannot run CARLA (needs a discrete GPU + ~20GB+
disk; CARLA is UE5-based). So instead of a real CARLA feed, this script
synthesizes what a CARLA-like sensor rig would hand the perception node:
a forward camera frame (road + grass + a pedestrian + a cone) and a lidar
point cloud with the same two obstacles, across a short sequence of frames
so the temporal filter has something to prove.

It runs the EXACT SAME modules the real node uses (segmentation, obstacles,
bev, temporal, occupancy) -- only the data source differs. Swap this script
for tools/carla_feed.py (real CARLA) or a real camera/lidar driver, and the
rest of the pipeline is unchanged; that's the whole point of the
sim-to-real architecture.

The pedestrian/cone pixel boxes are placed by projecting their TRUE ground
positions through the camera homography (world_to_pixel below), so the
camera-derived detection and the lidar point land on the same ground cell
-- exactly what a real, well-calibrated rig would give you. (An earlier
version of this script hand-picked pixel coordinates independently of the
lidar coordinates, which planted a third, spurious obstacle blob purely
from that inconsistency -- not a pipeline bug, but worth getting right so
the demo doesn't misrepresent the algorithm.)

Run:  PYTHONPATH=.. python3 tools/synthetic_demo.py
Output: demo_output/synthetic_demo.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from perception_costmap.occupancy import GridSpec, build_cost_array
from perception_costmap.bev import homography_from_camera, bev_known_mask
from perception_costmap.segmentation import create_segmenter
from perception_costmap.obstacles import (
    boxes_to_footprint_mask, points_to_grid_mask, camera_obstacle_mask_to_grid,
)
from perception_costmap.temporal import TemporalObstacleFilter


IMG_W, IMG_H = 640, 480
GRASS_BGR = (60, 130, 60)
ROAD_BGR = (110, 110, 110)
PEDESTRIAN_BGR = (30, 30, 30)
CONE_BGR = (0, 110, 255)


def world_to_pixel(x, y, H, grid):
    """Ground point (world x,y, metres) -> image pixel (u,v), by inverting
    the same homography bev.warp_to_bev uses (image px -> grid col,row).
    Mirrors bev.draw_grid_on_image's internal helper."""
    col = (x - grid.x_min) / grid.resolution
    row = (y - grid.y_min) / grid.resolution
    Hinv = np.linalg.inv(H)
    p = Hinv @ np.array([col, row, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return p[0] / p[2], p[1] / p[2]


def make_scene(ped_pixel, cone_pixel, pedestrian_visible=True, noise=0):
    """A synthetic forward-camera frame: green grass either side of a grey
    road, a dark pedestrian box standing in the road ahead, and an orange
    cone off to the right shoulder. Box bottoms are anchored at the given
    (already ground-projected) pixel positions so the visual detection and
    the "true" metric position agree. `noise` jitters the pedestrian box a
    couple pixels frame to frame (simulates real detector jitter)."""
    img = np.full((IMG_H, IMG_W, 3), GRASS_BGR, dtype=np.uint8)
    road_pts = np.array([[80, IMG_H], [IMG_W - 80, IMG_H],
                         [IMG_W // 2 + 60, IMG_H // 2],
                         [IMG_W // 2 - 60, IMG_H // 2]], dtype=np.int32)
    cv2.fillPoly(img, [road_pts], ROAD_BGR)

    rng = np.random.default_rng(0)
    jitter = rng.integers(-noise, noise + 1, size=2) if noise else (0, 0)

    ped_box = None
    if pedestrian_visible:
        px, py = ped_pixel
        cx, cy = int(px) + int(jitter[0]), int(py) + int(jitter[1])
        ped_box = (cx - 14, cy - 60, cx + 14, cy)     # bottom edge = ground point
        cv2.rectangle(img, ped_box[:2], ped_box[2:], PEDESTRIAN_BGR, -1)

    qx, qy = cone_pixel
    cone_box = (int(qx) - 16, int(qy) - 30, int(qx) + 16, int(qy))
    cv2.rectangle(img, cone_box[:2], cone_box[2:], CONE_BGR, -1)

    return img, ped_box, cone_box


def main():
    grid = GridSpec(x_min=-2.0, x_max=18.0, y_min=-8.0, y_max=8.0, resolution=0.1)
    K = np.array([[500, 0, IMG_W / 2], [0, 500, IMG_H / 2], [0, 0, 1]], dtype=np.float64)
    H = homography_from_camera(K, cam_height=1.6, pitch_deg=12, grid=grid)
    known_mask = bev_known_mask(H, (IMG_H, IMG_W), grid)

    segmenter = create_segmenter(
        "hsv",
        lower_hsv=(0, 0, 80), upper_hsv=(180, 40, 160),  # grey road, not green grass
        min_blob_area=2000,
    )

    temporal = TemporalObstacleFilter((grid.height, grid.width), hit=0.4, miss=0.2, threshold=0.5)

    # True metric position of the synthetic obstacles: pedestrian ~6.5m
    # ahead on the centerline, cone ~5m ahead, ~2.3m to the right. The
    # camera's visual boxes are placed by projecting THESE through H, so
    # camera and lidar agree on where the obstacle actually is.
    pedestrian_xyz = np.array([[6.5, 0.0, 0.9]])
    cone_xyz = np.array([[5.0, -2.3, 0.3]])
    ped_pixel = world_to_pixel(6.5, 0.0, H, grid)
    cone_pixel = world_to_pixel(5.0, -2.3, H, grid)

    frames = []
    # Frame 1-2: pedestrian visible and detected steadily -> should latch lethal.
    # Frame 3: pedestrian occluded for one frame (temporal filter should NOT
    # immediately clear it -- that's the point of requiring misses to clear).
    # Frame 4: pedestrian back -> stays lethal.
    schedule = [True, True, False, True]

    last_img = None
    last_road_mask = None
    for i, ped_visible in enumerate(schedule):
        img, ped_box, cone_box = make_scene(ped_pixel, cone_pixel,
                                            pedestrian_visible=ped_visible, noise=2)
        last_img = img

        road_img_mask = segmenter(img)
        road_grid_mask = camera_obstacle_mask_to_grid(road_img_mask, H, grid, known_mask=known_mask)
        last_road_mask = road_img_mask

        cam_boxes = [cone_box]
        if ped_visible:
            cam_boxes.append(ped_box)
        obstacle_img_mask = boxes_to_footprint_mask(cam_boxes, img.shape, footprint_frac=0.4)
        cam_obstacle_grid = camera_obstacle_mask_to_grid(obstacle_img_mask, H, grid, known_mask=known_mask)

        lidar_pts = cone_xyz if not ped_visible else np.vstack([pedestrian_xyz, cone_xyz])
        lidar_obstacle_grid = points_to_grid_mask(lidar_pts, grid)

        fused_raw = cam_obstacle_grid | lidar_obstacle_grid
        observed = known_mask | lidar_obstacle_grid
        fused_filtered = temporal.update(fused_raw, observed)

        # Per-class raw detections, gated by the temporal-confirmed mask: a
        # class's halo should only apply where that class's own raw
        # detection AND the temporal filter agree an obstacle is really
        # there. Feeding raw (unfiltered) per-class masks into
        # obstacle_layers would silently bypass the temporal filter, since
        # occupancy.build_cost_array applies each layer's halo unconditionally.
        pedestrian_raw = points_to_grid_mask(pedestrian_xyz, grid)
        cone_raw = points_to_grid_mask(cone_xyz, grid)
        cost = build_cost_array(
            grid, road_grid_mask, fused_filtered, known_mask=known_mask,
            road_edge_radius=1.5, unknown_infill=True,
            obstacle_layers={
                "person": dict(mask=pedestrian_raw & fused_filtered,
                               radius=2.5, scaling=1.5),
                "cone": dict(mask=cone_raw & fused_filtered,
                            radius=0.6, scaling=5.0),
            },
        )
        frames.append((i, ped_visible, cost.copy()))
        col, row = grid.world_to_cell(*pedestrian_xyz[0, :2])
        print(f"frame {i}: pedestrian_visible={ped_visible!s:5} "
             f"conf@ped_cell={temporal.conf[row, col]:.2f} "
             f"cost@ped_cell={cost[row, col]:3d}")

    render(last_img, last_road_mask, known_mask, frames, grid)
    print("\nwrote demo_output/synthetic_demo.png")


def render(cam_img, road_img_mask, known_mask, frames, grid):
    out_dir = os.path.join(os.path.dirname(__file__), "..", "demo_output")
    os.makedirs(out_dir, exist_ok=True)

    cmap = ListedColormap(["#dddddd", "#2ecc71", "#c9a13b", "#e74c3c"])
    bounds = [-1.5, -0.5, 40, 90, 100.5]
    norm = BoundaryNorm(bounds, cmap.N)

    n_frames = len(frames)
    fig, axes = plt.subplots(2, max(3, n_frames), figsize=(4 * max(3, n_frames), 8))

    axes[0, 0].imshow(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("synthetic camera frame (last frame)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(road_img_mask, cmap="gray")
    axes[0, 1].set_title("HSV road mask (image space)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(known_mask, cmap="gray", origin="lower")
    axes[0, 2].set_title("camera known-footprint (BEV)")
    axes[0, 2].axis("off")

    for j in range(3, max(3, n_frames)):
        axes[0, j].axis("off")

    for i, ped_visible, cost in frames:
        ax = axes[1, i]
        ax.imshow(cost, cmap=cmap, norm=norm, origin="lower")
        ax.set_title(f"frame {i} costmap, ped_visible={ped_visible}")
        ax.axis("off")

    title = ("Synthetic sim-to-real demo: grey=unknown, green=free/road, "
             "brown=off-road, red=lethal (obstacle + inflation halo). "
             "Frame 2 hides the pedestrian for one tick -- temporal filter "
             "keeps it lethal instead of clearing on a single miss.")
    fig.suptitle(title, fontsize=10, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(out_dir, "synthetic_demo.png"), dpi=130)


if __name__ == "__main__":
    main()
