"""scene: COCO-pretrained nano instance segmentation (ultralytics).

Masks only — boxes are produced by the backend but never used for drawing.
COCO ids -> our classes: person(0); vehicle(1,2,3,5,7); traffic_light(9);
traffic_sign(11 = stop sign).
"""

import numpy as np

from .base import SegModel

COCO_TO_CLASS = {
    0: "person",
    1: "vehicle", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle",
    9: "traffic_light",
    11: "traffic_sign",
}


class SceneModel(SegModel):
    name = "scene"

    def __init__(self, weights="models/yolo11n-seg.pt", conf=0.35,
                 device=None, imgsz=640):
        super().__init__()
        self.weights = weights
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def _load(self):
        from ultralytics import YOLO
        return YOLO(self.weights)

    def _predict(self, bgr):
        h, w = bgr.shape[:2]
        res = self._backend.predict(bgr, conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False,
                                    classes=sorted(COCO_TO_CLASS))[0]
        out = {}
        if res.masks is None:
            return out
        import cv2
        masks = res.masks.data.cpu().numpy()            # N x mh x mw
        clses = res.boxes.cls.cpu().numpy().astype(int)  # class ids only,
        for m, cid in zip(masks, clses):                 # never drawn as boxes
            name = COCO_TO_CLASS.get(int(cid))
            if name is None:
                continue
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            out[name] = out[name] | m if name in out else m
        return out
