#!/usr/bin/env python3
"""
ipm_overlay.py -- calibration sanity check, no ROS/CARLA required.

A bad IPM homography poisons the whole costmap and is invisible in RViz --
the road looks fine, obstacles look fine, they're just all in the wrong
place. This tool draws the metric grid (1m lines) back onto a single camera
frame so a human can check "that line really is 5m ahead" in seconds,
whether the frame came from CARLA or a phone photo of the real car's view.

Usage (points-mode calibration -- click 4 ground points yourself):
    python3 tools/ipm_overlay.py --image frame.png \
        --image-pts "100,480 540,480 220,300 420,300" \
        --world-pts "2,-1.5 2,1.5 8,-1.5 8,1.5" \
        --out overlay.png

Usage (camera-mode -- known intrinsics + mounting, e.g. CARLA):
    python3 tools/ipm_overlay.py --image frame.png \
        --fx 500 --fy 500 --cx 320 --cy 240 \
        --cam-height 1.6 --pitch 12 --yaw 0 \
        --out overlay.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from perception_costmap.occupancy import GridSpec
from perception_costmap.bev import (
    homography_from_points, homography_from_extrinsics, draw_grid_on_image,
)


def parse_pts(s):
    return [tuple(float(v) for v in pair.split(",")) for pair in s.split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="camera frame to overlay onto")
    ap.add_argument("--out", default="ipm_overlay_out.png")
    ap.add_argument("--spacing", type=float, default=1.0, help="grid line spacing, metres")
    ap.add_argument("--grid", default="-4,16,-10,10,0.1",
                    help="x_min,x_max,y_min,y_max,resolution")

    ap.add_argument("--image-pts", help="4 pixel points 'u,v u,v u,v u,v' (points mode)")
    ap.add_argument("--world-pts", help="4 ground points 'x,y x,y x,y x,y' metres (points mode)")

    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--cam-height", type=float, default=1.5)
    ap.add_argument("--cam-x", type=float, default=0.0)
    ap.add_argument("--cam-y", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=15.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    args = ap.parse_args()

    x_min, x_max, y_min, y_max, res = (float(v) for v in args.grid.split(","))
    grid = GridSpec(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, resolution=res)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")

    if args.image_pts and args.world_pts:
        H = homography_from_points(parse_pts(args.image_pts), parse_pts(args.world_pts), grid)
        print("mode: 4-point correspondence")
    elif args.fx and args.fy and args.cx and args.cy:
        K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=np.float64)
        H = homography_from_extrinsics(K, (args.cam_x, args.cam_y, args.cam_height),
                                       args.pitch, args.yaw, grid)
        print("mode: camera intrinsics + mounting")
    else:
        raise SystemExit("provide either --image-pts/--world-pts or --fx/--fy/--cx/--cy")

    overlay = draw_grid_on_image(img, H, grid, spacing_m=args.spacing)
    cv2.imwrite(args.out, overlay)
    print(f"wrote {args.out} -- check that grid lines land where you'd expect "
         f"a tape-measure mark at each labelled distance.")


if __name__ == "__main__":
    main()
