#!/usr/bin/env python3
"""
Per-stage timing of the perception pipeline on synthetic frames (or --image).
Run on any machine; the number that matters is the Jetson's.

    python3 tools/bench_perception.py --frames 50
    python3 tools/bench_perception.py --frames 50 --yolo-weights yolov8n.engine \
        --twinlite-repo TwinLiteNetPlus --twinlite-weights nano.pth
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.occupancy import GridSpec, build_cost_array
from perception_costmap import bev, obstacles
from perception_costmap.segmentation import create_segmenter


def timed(fn, frames):
    t0 = time.perf_counter()
    for f in frames:
        out = fn(f)
    dt = (time.perf_counter() - t0) / len(frames)
    return out, dt * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--image", default=None)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--twinlite-repo", default=None)
    ap.add_argument("--twinlite-weights", default=None)
    args = ap.parse_args()

    if args.image:
        base = cv2.imread(args.image)
    else:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (360, 640, 3), np.uint8).astype(np.uint8)
        base[200:, :] = (90, 90, 90)      # bottom half "asphalt"
    frames = [base.copy() for _ in range(args.frames)]

    grid = GridSpec()
    H = bev.homography_from_points(
        [(0, 200), (640, 200), (640, 360), (0, 360)],
        [(16, 10), (16, -10), (3, -3), (3, 3)], grid)

    rows = []
    seg = create_segmenter("hsv")
    road, ms = timed(seg, frames)
    rows.append(("segment (hsv)", ms))

    if args.twinlite_repo and args.twinlite_weights:
        tl = create_segmenter("twinlitenet", repo_path=args.twinlite_repo,
                              weights=args.twinlite_weights)
        _, ms = timed(tl, frames)
        rows.append(("segment (twinlitenet)", ms))

    _, ms = timed(lambda f: obstacles.detect_obstacles_camera(f, road), frames)
    rows.append(("obstacles (classical)", ms))

    if args.yolo_weights:
        det = obstacles.YoloObstacleDetector(weights=args.yolo_weights)
        _, ms = timed(det.detect, frames)
        rows.append(("obstacles (yolo)", ms))

    road_u8 = road.astype(np.uint8) * 255
    _, ms = timed(lambda f: bev.warp_to_bev(road_u8, H, grid), frames)
    rows.append(("bev warp", ms))

    road_bev = bev.warp_to_bev(road_u8, H, grid) > 127
    obst = np.zeros_like(road_bev)
    _, ms = timed(lambda f: build_cost_array(grid, road_bev, obst), frames)
    rows.append(("build cost array", ms))

    total = sum(ms for _, ms in rows)
    print("\n%-24s %8s" % ("stage", "ms/frame"))
    for name, ms in rows:
        print("%-24s %8.2f" % (name, ms))
    print("%-24s %8.2f  (~%.1f Hz worst-case serial)" % ("TOTAL", total, 1000.0 / total))


if __name__ == "__main__":
    main()
