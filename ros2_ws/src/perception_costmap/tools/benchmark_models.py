#!/usr/bin/env python3
"""Benchmark deployed road and object models on representative RGB frames."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.obstacles import YoloObstacleDetector
from perception_costmap.segmentation import create_segmenter


def timed_call(callable_, image, warmup=2):
    for _ in range(warmup):
        callable_(image)
    started = time.perf_counter()
    result = callable_(image)
    return result, 1000.0 * (time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--twinlite-repo", required=True)
    parser.add_argument("--twinlite-weights", required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    images = [(Path(path), cv2.imread(path)) for path in args.images]
    if any(image is None for _, image in images):
        raise SystemExit("one or more images could not be read")
    hsv = create_segmenter("hsv")
    twin = create_segmenter(
        "twinlitenet", repo_path=args.twinlite_repo,
        weights=args.twinlite_weights, config="nano")
    yolo = YoloObstacleDetector(args.yolo_weights, conf=0.35)

    report = {"images": []}
    for path, image in images:
        record = {"image": str(path)}
        for name, model in (("hsv", hsv), ("twinlitenet", twin)):
            mask, elapsed = timed_call(model, image)
            record[name] = {
                "latency_ms": elapsed,
                "road_fraction": float(np.mean(mask)),
            }
        groups, elapsed = timed_call(yolo.detect_grouped, image)
        record["yolo"] = {
            "latency_ms": elapsed,
            "groups": {name: int(mask.sum()) for name, mask in groups.items()},
        }
        report["images"].append(record)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
