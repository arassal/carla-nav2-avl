"""course: cones + white lines -- the classes no pretrained COCO model has.

Hybrid, two backends:
  cones      ExStella/Traffic-cones (HF, Apache-2.0): a YOLOv8s detector
             trained on a dedicated cone dataset -- catches striped and
             white-banded cones our color-gate-taught model misses. Boxes
             are converted to AREA masks (color gate inside each box, convex
             hull fallback) -- no bounding boxes are ever rendered.
  white_line our fine-tuned yolo11n-seg (tools/train_course.py).

Either backend degrades independently (base class warns once).
"""

import numpy as np

from .base import SegModel

ID_TO_CLASS = {0: "cone", 1: "white_line"}


def _box_to_cone_mask(bgr, x1, y1, x2, y2):
    """Area mask for one detected cone: orange/pale/white gate inside the
    box, convex-hulled; falls back to the box's lower 85% if the gate finds
    nothing (dark/odd-colored cone)."""
    import cv2
    h, w = bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (2, 90, 70), (22, 255, 255))          # orange
    m |= cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))         # white bands
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    out = np.zeros((h, w), bool)
    if m.mean() > 0.10 * 255:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in cnts]))
        mm = np.zeros(m.shape, np.uint8)
        cv2.fillPoly(mm, [hull], 255)
        out[y1:y2, x1:x2] = mm > 0
    else:                                   # gate failed: lower box region
        yy = y1 + int(0.15 * (y2 - y1))
        out[yy:y2, x1:x2] = True
    return out


class CourseModel(SegModel):
    name = "course"

    def __init__(self, weights="models/course.pt",
                 cone_weights="models/cone_det.pt",
                 conf=0.30, cone_conf=0.35, device=None, imgsz=640):
        super().__init__()
        self.weights = weights
        self.cone_weights = cone_weights
        self.conf = conf
        self.cone_conf = cone_conf
        self.device = device
        self.imgsz = imgsz

    def _load(self):
        from ultralytics import YOLO
        import os
        seg = YOLO(self.weights) if os.path.exists(self.weights) else None
        det = YOLO(self.cone_weights) if os.path.exists(self.cone_weights) else None
        if seg is None and det is None:
            raise FileNotFoundError("%s and %s both missing" %
                                    (self.weights, self.cone_weights))
        return (seg, det)

    def _predict(self, bgr):
        import cv2
        seg, det = self._backend
        h, w = bgr.shape[:2]
        out = {}

        if det is not None:                 # cones: detector -> area masks
            r = det.predict(bgr, conf=self.cone_conf, imgsz=self.imgsz,
                            device=self.device, verbose=False)[0]
            for b in r.boxes.xyxy.cpu().numpy():
                m = _box_to_cone_mask(bgr, *b[:4])
                if m is not None:
                    out["cone"] = out["cone"] | m if "cone" in out else m

        if seg is not None:                 # white_line (+ cone fallback)
            r = seg.predict(bgr, conf=self.conf, imgsz=self.imgsz,
                            device=self.device, verbose=False)[0]
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy().astype(int)
                for m, cid in zip(masks, clses):
                    name = ID_TO_CLASS.get(int(cid))
                    if name is None:
                        continue
                    if name == "cone" and det is not None:
                        continue            # detector owns cones when present
                    m = cv2.resize(m, (w, h),
                                   interpolation=cv2.INTER_LINEAR) > 0.5
                    out[name] = out[name] | m if name in out else m
        return out
