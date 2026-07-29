"""road: YOLOPv2 dual-head segmenter -> road (drivable area) + lane_line.

Backend: the official TorchScript checkpoint (CAIC-AD/YOLOPv2 release).
We use only the two segmentation heads; the detection head is ignored.
"""

import numpy as np

from .base import SegModel


def _letterbox(img, new=(384, 640)):
    import cv2
    h, w = img.shape[:2]
    r = min(new[0] / h, new[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top = (new[0] - nh) // 2
    left = (new[1] - nw) // 2
    out = np.full((new[0], new[1], 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out, r, top, left, nh, nw


class RoadModel(SegModel):
    name = "road"

    def __init__(self, weights="models/yolopv2.pt", device=None,
                 input_hw=(384, 640)):
        super().__init__()
        self.weights = weights
        self.input_hw = input_hw
        self.device = device

    def _load(self):
        import torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.jit.load(self.weights, map_location=dev)
        model = model.to(dev).eval()
        if dev == "cuda":
            model = model.half()
        self._dev = dev
        self._torch = torch
        return model

    def _predict(self, bgr):
        import cv2
        torch = self._torch
        h, w = bgr.shape[:2]
        img, r, top, left, nh, nw = _letterbox(bgr, self.input_hw)
        t = torch.from_numpy(img[:, :, ::-1].copy()).to(self._dev)
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = (t.half() if self._dev == "cuda" else t.float()) / 255.0
        with torch.no_grad():
            _, seg, ll = self._backend(t)      # (det, drivable, lane)
        out = {}
        for name, head, thresh in (("road", seg, None), ("lane_line", ll, 0.5)):
            m = head.float().squeeze(0).cpu().numpy()
            if m.ndim == 3 and m.shape[0] == 2:      # 2-ch: argmax
                m = m[1] > m[0]
            else:
                m = m.squeeze() > (thresh if thresh is not None else 0.5)
            m = m[top:top + nh, left:left + nw]      # un-letterbox
            m = cv2.resize(m.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            out[name] = m
        # lanes win where both heads fire
        out["road"] = out["road"] & ~out["lane_line"]
        return out
