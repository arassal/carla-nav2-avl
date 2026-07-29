#!/usr/bin/env python3
"""
Export YOLOv8 to a TensorRT FP16 engine. RUN THIS ON THE JETSON — a .engine
built on the 5090 will not load on the Orin. Afterwards set
yolo_weights: /path/yolov8n.engine in the YAML; the detector class loads
either format.

    python3 tools/export_trt.py --weights yolov8n.pt --imgsz 640
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO
    path = YOLO(args.weights).export(format="engine", half=True, imgsz=args.imgsz)
    print("engine written: %s" % path)


if __name__ == "__main__":
    main()
