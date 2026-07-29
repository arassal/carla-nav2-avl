#!/usr/bin/env python3
"""
Overlay the costmap's metric grid onto a camera frame to eyeball the IPM.

    python3 tools/ipm_overlay.py --image frame.png \
        --config config/perception_costmap.yaml --camera front --out overlay.png

Reads the same YAML the node uses, builds the same homography, draws 1 m grid
lines. If the 5 m line isn't 5 m away in the scene, fix the calibration
BEFORE debugging anything downstream. ROS not required.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.occupancy import GridSpec
from perception_costmap import bev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--out", default="ipm_overlay.png")
    args = ap.parse_args()

    params = yaml.safe_load(open(args.config))["perception_costmap"]["ros__parameters"]
    grid = GridSpec(x_min=params["x_min"], x_max=params["x_max"],
                    y_min=params["y_min"], y_max=params["y_max"],
                    resolution=params["resolution"])
    cam = params[args.camera]
    img = cv2.imread(args.image)
    if img is None:
        sys.exit("could not read %s" % args.image)

    if cam.get("ipm_mode", "points") == "points":
        H = bev.homography_from_points(
            np.array(cam["ipm_image_pts"], float).reshape(4, 2),
            np.array(cam["ipm_world_pts"], float).reshape(4, 2), grid)
    else:
        sys.exit("camera mode needs a live camera_info; use ipm_mode: points here")

    cv2.imwrite(args.out, bev.draw_grid_on_image(img, H, grid))
    print("wrote %s -- check that the labelled distances match the scene" % args.out)


if __name__ == "__main__":
    main()
