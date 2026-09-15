#!/usr/bin/env python3
"""Score a camera IPM against surveyed image/ground correspondences."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap import bev
from perception_costmap.evaluation import projection_errors
from perception_costmap.occupancy import GridSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help="JSON calibration sample file")
    parser.add_argument("--max-rmse", type=float, default=0.25)
    args = parser.parse_args()
    data = json.loads(Path(args.samples).read_text())
    grid = GridSpec(**data.get("grid", {}))
    H = bev.homography_from_extrinsics(
        data["K"], data["cam_xyz"], data["pitch_deg"], data["yaw_deg"], grid)
    errors = projection_errors(
        H, [s["pixel"] for s in data["samples"]],
        [s["ground_xy"] for s in data["samples"]], grid)
    report = {
        "samples": len(errors),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "mean_m": float(np.mean(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "pass": bool(np.sqrt(np.mean(errors ** 2)) <= args.max_rmse),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
