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
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.carla_convert import semantic_to_road_mask, mask_iou
from perception_costmap.segmentation import create_segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--road-tags", type=int, nargs="+", default=[1, 24])
    ap.add_argument("--twinlite-repo", default=None)
    ap.add_argument("--twinlite-weights", default=None)
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
    for rgb_path in pairs:
        sem_path = Path(str(rgb_path).replace("_rgb.png", "_sem.png"))
        rgb, sem = cv2.imread(str(rgb_path)), cv2.imread(str(sem_path))
        if rgb is None or sem is None:
            continue
        truth = semantic_to_road_mask(sem, tuple(args.road_tags))
        for name, seg in segmenters.items():
            scores[name].append(mask_iou(seg(rgb), truth))

    print("\nroad-mask IoU vs CARLA semantic truth (%d frames):" % len(pairs))
    for name, vals in sorted(scores.items()):
        print("  %-12s mean %.3f   min %.3f" % (name, np.mean(vals), np.min(vals)))
    best = max(scores, key=lambda n: np.mean(scores[n]))
    print("winner: %s -> set segmentation_method: %s in the YAML" % (best, best))


if __name__ == "__main__":
    main()
