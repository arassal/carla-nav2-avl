#!/usr/bin/env python3
"""
bench_perception.py -- per-stage timing on whatever machine you run it on
(dev laptop, CARLA sim box, Jetson). Numbers do NOT transfer between
machines (different CPU/GPU, presence/absence of TensorRT) -- always
re-run on the actual target before trusting a latency claim.

Usage:
    python3 tools/bench_perception.py --frames 20
    python3 tools/bench_perception.py --frames 50 --yolo-weights yolov8n.engine
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from perception_costmap.occupancy import GridSpec, build_cost_array
from perception_costmap.bev import homography_from_camera, bev_known_mask, warp_to_bev
from perception_costmap.segmentation import create_segmenter
from perception_costmap.obstacles import (
    points_to_grid_mask, detect_obstacles_classical, camera_obstacle_mask_to_grid,
    YoloObstacleDetector,
)
from perception_costmap.temporal import TemporalObstacleFilter


def timed(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0  # ms/call


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--yolo-weights", default=None, help="if set, benchmark YOLO instead of the classical detector")
    ap.add_argument("--twinlite-weights", default=None, help="if set, benchmark TwinLiteNet+ instead of HSV")
    args = ap.parse_args()

    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    grid = GridSpec(x_min=-4, x_max=16, y_min=-10, y_max=10, resolution=0.1)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    H = homography_from_camera(K, cam_height=1.6, pitch_deg=12, grid=grid)
    known_mask = bev_known_mask(H, img.shape, grid)

    if args.twinlite_weights:
        segmenter = create_segmenter("twinlitenet", weights_path=args.twinlite_weights)
        seg_label = "segment(twinlitenet)" + (" [FALLBACK->hsv]" if segmenter.using_fallback else "")
    else:
        segmenter = create_segmenter("hsv")
        seg_label = "segment(hsv)"

    if args.yolo_weights:
        yolo = YoloObstacleDetector(weights=args.yolo_weights)
        yolo.detect(img)  # trigger warm-load once before labeling
        obs_label = "obstacles(yolo)" + (" [load failed -> would fall back]" if yolo._load_failed else "")
    else:
        yolo = None
        obs_label = "obstacles(classical)"

    temporal = TemporalObstacleFilter((grid.height, grid.width))
    lidar_pts = np.random.default_rng(1).uniform(-5, 5, size=(2000, 3))

    rows = []
    rows.append((seg_label, timed(lambda: segmenter(img), args.frames)))
    if yolo is not None:
        rows.append((obs_label, timed(lambda: yolo.detect(img), args.frames)))
    else:
        rows.append((obs_label, timed(lambda: detect_obstacles_classical(img), args.frames)))
    rows.append(("lidar binning", timed(lambda: points_to_grid_mask(lidar_pts, grid), args.frames)))
    rows.append(("bev warp (road mask)", timed(
        lambda: camera_obstacle_mask_to_grid(np.ones(img.shape[:2], dtype=np.uint8), H, grid, known_mask=known_mask),
        args.frames)))

    road = np.ones((grid.height, grid.width), dtype=bool)
    obstacle = np.zeros((grid.height, grid.width), dtype=bool)
    rows.append(("temporal filter update", timed(lambda: temporal.update(obstacle, known_mask), args.frames)))
    rows.append(("build_cost_array", timed(
        lambda: build_cost_array(grid, road, obstacle, known_mask=known_mask,
                                 road_edge_radius=1.5, unknown_infill=True),
        args.frames)))

    print(f"\n{args.frames} iterations, grid {grid.height}x{grid.width} @ {grid.resolution}m/cell\n")
    print(f"{'stage':<28}{'ms/frame':<12}{'Hz (this stage alone)':<10}")
    total_ms = 0.0
    for name, ms in rows:
        hz = 1000.0 / ms if ms > 0 else float("inf")
        print(f"{name:<28}{ms:<12.3f}{hz:<10.1f}")
        total_ms += ms
    print(f"{'TOTAL (serial sum)':<28}{total_ms:<12.3f}{1000.0/total_ms:<10.1f}  Hz")


if __name__ == "__main__":
    main()
