#!/usr/bin/env python3
"""
Score road segmenters against CARLA semantic ground truth. Offline: feed it
the paired PNGs from carla_feed.py --dump-dir. Prints per-method mean IoU in
image space (does the mask match the road?) and reports the winner.

    python3 tools/eval_road_iou.py --pairs /tmp/pairs \
        [--twinlite-repo TwinLiteNetPlus --twinlite-weights nano.pth]
    # add --road-tags if the printed tag list from carla_feed says 1/24 is wrong
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.carla_convert import semantic_to_road_mask
from perception_costmap.segmentation import create_segmenter
from perception_costmap.evaluation import binary_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--road-tags", type=int, nargs="+", default=[1, 24])
    ap.add_argument("--twinlite-repo", default=None)
    ap.add_argument("--twinlite-weights", default=None)
    ap.add_argument("--truth-suffix", default="_sem.png",
                    help="_sem.png for CARLA tags or _mask.png for binary labels")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    segmenters = {"hsv": create_segmenter("hsv")}
    if args.twinlite_repo and args.twinlite_weights:
        try:
            segmenters["twinlitenet"] = create_segmenter(
                "twinlitenet", repo_path=args.twinlite_repo,
                weights=args.twinlite_weights)
        except Exception as e:
            print("twinlitenet unavailable: %s" % e)

    pairs = sorted(Path(args.pairs).glob("*_rgb.png"))
    if not pairs:
        sys.exit("no *_rgb.png in %s" % args.pairs)

    scores = {name: [] for name in segmenters}
    latencies = {name: [] for name in segmenters}
    for rgb_path in pairs:
        sem_path = Path(str(rgb_path).replace("_rgb.png", args.truth_suffix))
        rgb, sem = cv2.imread(str(rgb_path)), cv2.imread(str(sem_path))
        if rgb is None or sem is None:
            continue
        if args.truth_suffix == "_sem.png":
            truth = semantic_to_road_mask(sem, tuple(args.road_tags))
        else:
            truth = cv2.cvtColor(sem, cv2.COLOR_BGR2GRAY) >= 128
        for name, seg in segmenters.items():
            started = time.perf_counter()
            prediction = seg(rgb)
            latencies[name].append(1000.0 * (time.perf_counter() - started))
            scores[name].append(binary_metrics(prediction, truth))

    print("\nroad-mask IoU vs CARLA semantic truth (%d frames):" % len(pairs))
    report = {"frames": len(pairs), "methods": {}}
    for name, vals in sorted(scores.items()):
        metrics = {
            key: float(np.mean([v[key] for v in vals]))
            for key in ("iou", "precision", "recall", "f1")
        }
        metrics["latency_mean_ms"] = float(np.mean(latencies[name]))
        metrics["latency_p95_ms"] = float(np.percentile(latencies[name], 95))
        report["methods"][name] = metrics
        print("  %-12s IoU %.3f F1 %.3f recall %.3f p95 %.1f ms" % (
            name, metrics["iou"], metrics["f1"], metrics["recall"],
            metrics["latency_p95_ms"]))
    best = max(scores, key=lambda n: report["methods"][n]["iou"])
    report["best_iou"] = best
    print("best measured IoU: %s" % best)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
