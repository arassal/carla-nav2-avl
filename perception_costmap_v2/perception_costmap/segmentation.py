"""
segmentation.py — road-mask backends behind one callable interface, selected
by a factory so costmap_node.py (and tools/eval_road_iou.py) can switch
methods purely through config.

  create_segmenter("hsv", **kw)         -> HsvSegmenter instance (callable)
  create_segmenter("twinlitenet", **kw) -> TwinLiteSegmenter instance

Both are callables: `mask = segmenter(img_bgr)`.

torch stays a lazy import -- HsvSegmenter never touches it, and
TwinLiteSegmenter only imports it at construction time, with a fallback to
HSV if the model/weights fail to load. A missing model must never take the
perception node down.
"""

import numpy as np
import cv2


def letterbox(img, new_size=640):
    """
    Resize+pad to a square `new_size` x `new_size` canvas preserving aspect
    ratio (the standard YOLO/segmentation preprocessing trick), returning
    the transform needed to map model-space coordinates back to the
    original image.

    Returns (padded_img, ratio, (pad_left, pad_top)).
    """
    h, w = img.shape[:2]
    ratio = min(new_size / h, new_size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = new_size - nh, new_size - nw
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    channels = () if img.ndim == 2 else (img.shape[2],)
    padded = np.zeros((new_size, new_size) + channels, dtype=img.dtype)
    padded[pad_top:pad_top + nh, pad_left:pad_left + nw, ...] = resized
    return padded, ratio, (pad_left, pad_top)


def unletterbox_mask(mask, ratio, pad, original_shape):
    """Inverse of letterbox: crop the padding off a model-space mask and
    resize back to the original image's (h, w). Output cropped to content
    extent so a learned mask never claims pixels the model never saw."""
    pad_left, pad_top = pad
    h, w = original_shape[:2]
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    cropped = mask[pad_top:pad_top + nh, pad_left:pad_left + nw]
    return cv2.resize(cropped.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


class HsvSegmenter:
    """
    Threshold + largest-connected-blob road segmenter (reused from Adam
    Castillo's original `perception/costmap.py` prototype). Lighting
    dependent -- see DESIGN.md known TODOs -- but has zero external model
    dependencies and never fails to load, so it is also the universal
    fallback for TwinLiteSegmenter.
    """

    def __init__(self, lower_hsv=(0, 0, 60), upper_hsv=(180, 60, 200),
                min_blob_area=500):
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)
        self.min_blob_area = min_blob_area

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        raw = cv2.inRange(hsv, self.lower, self.upper) > 0
        return self._largest_blob(raw)

    def _largest_blob(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return np.zeros_like(mask, dtype=bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        if areas[best - 1] < self.min_blob_area:
            return np.zeros_like(mask, dtype=bool)
        return labels == best


class TwinLiteSegmenter:
    """
    TwinLiteNet+ (nano, ~34K params, BDD100K drivable-area) adapter. Loads
    the model ONCE at construction. On any load failure (missing repo,
    missing weights, no torch, wrong device) it silently becomes an
    HsvSegmenter instead -- a broken learned-model config must degrade the
    accuracy, not take the node down.
    """

    def __init__(self, weights_path="nano.pth", input_size=640,
                device=None, fallback: "HsvSegmenter" = None):
        self.input_size = input_size
        self.device = device
        self._fallback = fallback or HsvSegmenter()
        self._model = None
        self._using_fallback = False
        self._load(weights_path)

    def _load(self, weights_path):
        try:
            import torch
            # Model class recipe lives in perception/twinLiteNetTest.py per
            # DESIGN.md -- import lazily so this module never requires torch
            # or that external repo just to be imported.
            from twinlitenetplus.model import TwinLiteNetPlus  # type: ignore
            model = TwinLiteNetPlus()
            state = torch.load(weights_path, map_location=device or "cpu")
            model.load_state_dict(state)
            model.eval()
            if device:
                model = model.to(device)
            self._model = model
            self._torch = torch
        except Exception:
            self._using_fallback = True

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if self._using_fallback or self._model is None:
            return self._fallback(img_bgr)
        try:
            padded, ratio, pad = letterbox(img_bgr, self.input_size)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = self._torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
            if self.device:
                tensor = tensor.to(self.device)
            with self._torch.no_grad():
                out = self._model(tensor)
            drivable = out[0] if isinstance(out, (tuple, list)) else out
            mask_model_space = (drivable.squeeze().cpu().numpy() > 0.5)
            return unletterbox_mask(mask_model_space, ratio, pad, img_bgr.shape)
        except Exception:
            # Inference-time failure (bad frame, OOM) -- degrade gracefully
            # for this frame only; don't flip using_fallback permanently.
            return self._fallback(img_bgr)


def create_segmenter(method: str = "hsv", **kw):
    """Factory: method in {"hsv", "twinlitenet"}. Unknown method raises
    (fail loud at config time, not silently at runtime)."""
    method = method.lower()
    if method == "hsv":
        return HsvSegmenter(**kw)
    if method in ("twinlitenet", "twinlite", "twinlitenetplus"):
        return TwinLiteSegmenter(**kw)
    raise ValueError(f"unknown segmentation_method: {method!r}")


def segment_road(img_bgr, **kw):
    """Back-compat function form: builds a fresh HsvSegmenter per call.
    Prefer `create_segmenter('hsv', ...)` once and reuse it -- this exists
    only so old call sites keep working."""
    return HsvSegmenter(**kw)(img_bgr)
