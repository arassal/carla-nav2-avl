#!/usr/bin/env python3
"""
eval_road_iou.py -- measured accuracy, not claimed. Scores road segmenters
(hsv vs twinlitenet) against CARLA's free semantic-camera ground truth.

Consumes paired RGB/semantic PNGs dumped by tools/carla_feed.py
(--dump-dir rgb_NNNNNN.png / sem_NNNNNN.png pairs). Runs entirely offline
on already-captured frames -- no live CARLA connection needed here, so
this half of the tool works on any machine, though the frames themselves
can only be captured on the sim box.

Reports per-frame and mean IoU in TWO spaces:
  - image space:  the segmenter's mask vs the semantic mask, as captured
  - BEV space:    both warped through the same homography used by the
                  real pipeline, since a segmenter can look fine in the
                  camera view and still be wrong where it actually
                  matters -- on the ground plane the costmap uses.

Usage:
    python3 tools/eval_road_iou.py --dump-dir /tmp/carla_eval_frames \
        --methods hsv twinlitenet --road-tags 1 24
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from perception_costmap.carla_convert import semantic_to_road_mask, mask_iou
from perception_costmap.segmentation import create_segmenter
from perception_costmap.occupancy import GridSpec
from perception_costmap.bev import homography_from_extrinsics, warp_to_bev


def find_pairs(dump_dir):
    rgb_paths = sorted(glob.glob(os.path.join(dump_dir, "rgb_*.png")))
    pairs = []
    for rgb_path in rgb_paths:
        idx = os.path.basename(rgb_path)[4:-4]
        sem_path = os.path.join(dump_dir, f"sem_{idx}.png")
        if os.path.exists(sem_path):
            pairs.append((rgb_path, sem_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--methods", nargs="+", default=["hsv"], help="segmentation_method values to compare")
    ap.add_argument("--road-tags", nargs="+", type=int, default=[1, 24],
                    help="CARLA semantic tag ids counted as road; PRINT AND VERIFY against "
                         "carla_feed's first-frame diagnostic before trusting the default")
    # Defaults match tools/carla_feed.py's camera (640x480, FOV 90, no
    # Rotation, mounted at x=1.5). fx=500/pitch=12 were placeholders and
    # scored the BEV column against a homography no frame was ever taken
    # with -- see DEPLOY.md section 3, and pass your own values for any
    # other camera.
    ap.add_argument("--cam-height", type=float, default=1.9,
                    help="camera height ABOVE THE ROAD, not the mount offset")
    ap.add_argument("--cam-x", type=float, default=1.5)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--fx", type=float, default=320.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    args = ap.parse_args()

    pairs = find_pairs(args.dump_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit(f"no rgb_*/sem_* pairs found in {args.dump_dir} "
                         f"-- run tools/carla_feed.py --dump-dir first")

    segmenters = {m: create_segmenter(m) for m in args.methods}

    sample_img = cv2.imread(pairs[0][0])
    h, w = sample_img.shape[:2]
    K = np.array([[args.fx, 0, w / 2], [0, args.fx, h / 2], [0, 0, 1]], dtype=np.float64)
    grid = GridSpec(x_min=0.0, x_max=20.0, y_min=-10.0, y_max=10.0, resolution=0.1)
    H = homography_from_extrinsics(K, (args.cam_x, 0.0, args.cam_height),
                                   args.pitch, 0.0, grid)

    results = {m: {"image": [], "bev": []} for m in args.methods}

    for rgb_path, sem_path in pairs:
        rgb = cv2.imread(rgb_path)
        sem = cv2.imread(sem_path)
        if rgb is None or sem is None or rgb.shape[:2] != sem.shape[:2]:
            continue
        gt_img = semantic_to_road_mask(sem, road_tags=tuple(args.road_tags))
        gt_bev = warp_to_bev(gt_img.astype(np.uint8), H, grid).astype(bool)

        for name, seg in segmenters.items():
            pred_img = seg(rgb)
            pred_bev = warp_to_bev(pred_img.astype(np.uint8), H, grid).astype(bool)
            results[name]["image"].append(mask_iou(pred_img, gt_img))
            results[name]["bev"].append(mask_iou(pred_bev, gt_bev))

    print(f"\n{len(pairs)} frame pairs from {args.dump_dir}\n")
    print(f"{'method':<14}{'IoU (image space)':<20}{'IoU (BEV space)':<20}")
    for name in args.methods:
        img_scores = results[name]["image"]
        bev_scores = results[name]["bev"]
        if not img_scores:
            print(f"{name:<14}{'no valid frames':<20}")
            continue
        print(f"{name:<14}{np.mean(img_scores):<20.4f}{np.mean(bev_scores):<20.4f}")

    print("\nBEV IoU is the number that matters for the actual costmap -- a "
         "segmenter can look fine in the camera view and still be wrong "
         "on the ground plane the planner uses.")


if __name__ == "__main__":
    main()
