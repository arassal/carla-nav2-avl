"""Fine-tune the course model (cone + white_line) from datasets/course.

    python3 tools/train_course.py --epochs 60
    -> models/course.pt

Thin elongated masks (painted lines) lose detail at the default mask
downsample: overlap_mask=False + mask_ratio=1 keep them crisp.
"""

import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/course/course.yaml")
    ap.add_argument("--base", default="models/yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project="runs", name="course",
                exist_ok=True, overlap_mask=False, mask_ratio=1,
                degrees=8.0, fliplr=0.5, mosaic=0.8, close_mosaic=10)
    best = os.path.join("runs", "course", "weights", "best.pt")
    shutil.copy(best, os.path.join("models", "course.pt"))
    print("saved models/course.pt")
    metrics = model.val(data=args.data)
    print("val mask mAP50:", metrics.seg.map50)


if __name__ == "__main__":
    main()
