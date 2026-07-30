#!/usr/bin/env python3
"""
export_trt.py -- YOLOv8 .pt -> TensorRT .engine.

RUN THIS ON THE TARGET DEVICE (e.g. Jetson) -- TensorRT engines are
hardware-specific; an engine built on a dev laptop/desktop will not load on
different silicon. After exporting, point the node at the .engine file:
YoloObstacleDetector(weights="yolov8n.engine") loads it transparently
through the same ultralytics API used for .pt.

Usage:
    python3 tools/export_trt.py --weights yolov8n.pt --imgsz 640 --half
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", default=True, help="FP16 (default on; matches Jetson Orin's strength)")
    ap.add_argument("--no-half", dest="half", action="store_false")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not importable. On the Jetson: NEVER `pip install "
            "torch` (pulls a CPU wheel) -- use NVIDIA's Jetson-matched wheel "
            "(developer.nvidia.com/embedded -> PyTorch for Jetson), then "
            "`pip install ultralytics --no-deps` plus its light deps."
        )

    model = YOLO(args.weights)
    path = model.export(format="engine", half=args.half, imgsz=args.imgsz)
    print(f"wrote {path}")
    print(f"set yolo_weights: {path} in config/perception_costmap.yaml")


if __name__ == "__main__":
    main()
