"""
segmentation.py
---------------
Drivable-road segmentation in image space. Returns a boolean mask where True
means "this pixel is road".

The classical HSV method is factored from Adam Castillo's
``perception/costmap.py`` (HSV threshold for low-saturation asphalt + keep the
largest connected blob). A learned segmenter (TwinLiteNet+) can be dropped in
behind the same ``segment_road`` interface without touching callers.
"""

import numpy as np
import cv2


def segment_road_hsv(img_bgr,
                     max_sat: int = 60,
                     val_lo: int = 40,
                     val_hi: int = 200) -> np.ndarray:
    """
    Classical road mask: asphalt is low-saturation, mid-brightness. Threshold
    in HSV, clean up, then keep only the largest connected blob (the road).
    Returns a boolean mask the size of the image.

    HSV ranges are lighting-dependent -- recalibrate for the real camera, or
    switch to the learned segmenter.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s < max_sat) & (v > val_lo) & (v < val_hi)).astype(np.uint8) * 255

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask > 0


def letterbox(img, new_size=640, pad_color=(114, 114, 114)):
    """Aspect-preserving resize + gray padding to new_size x new_size.
    (Same scheme as perception/twinLiteNetTest.py / the YOLO family.)"""
    h, w = img.shape[:2]
    ratio = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pad_w, pad_h = (new_size - new_w) / 2, (new_size - new_h) / 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_color)
    return padded, ratio, (left, top)


class HsvSegmenter:
    """Classical fallback -- zero deps, calibrate HSV ranges per camera."""
    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, img_bgr):
        return segment_road_hsv(img_bgr, **self.kw)


class TwinLiteSegmenter:
    """
    TwinLiteNet+ drivable-area head as a road segmenter.

    Setup (once): clone https://github.com/chequanghuy/TwinLiteNetPlus and
    download nano.pth -- full instructions in perception/twinLiteNetTest.py.
    Loads the network ONCE at construction; __call__ is inference only.
    """
    def __init__(self, repo_path, weights, config="nano", img_size=640,
                 device=None):
        import sys, argparse
        import torch                       # lazy: optional dependency
        self.torch = torch
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from model.model import TwinLiteNetPlus   # from the cloned repo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = TwinLiteNetPlus(argparse.Namespace(config=config))
        model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model = model.to(self.device).eval()
        self.img_size = img_size

    def __call__(self, img_bgr):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        padded, ratio, (pl, pt) = letterbox(img_bgr, self.img_size)
        with torch.no_grad():
            t = torch.from_numpy(padded).to(self.device).float()
            t = t.permute(2, 0, 1).unsqueeze(0) / 255.0
            da_out, _ = self.model(t)          # (drivable-area, lanes)
        # crop by content extent: letterbox padding can be asymmetric by 1px
        new_h = int(round(h * ratio))
        new_w = int(round(w * ratio))
        da = da_out[:, :, pt:pt + new_h, pl:pl + new_w]
        da = torch.nn.functional.interpolate(da, size=(h, w), mode="bilinear")
        return (torch.argmax(da, dim=1).squeeze(0).cpu().numpy() == 1)


class TwinLiteTRTSegmenter:
    """TwinLiteNet drivable-area head from a TensorRT engine (build:
    trtexec --onnx=twinlite_nano.onnx --fp16 --saveEngine=...). Fixed
    384x640 input with aspect-preserving letterbox; I/O buffers are torch
    CUDA tensors so there is no pycuda dependency. ~10x the pytorch path
    on the Jetson."""

    IN_H, IN_W = 384, 640

    def __init__(self, weights, **_ignored):
        import tensorrt as trt              # lazy: device-only dependency
        import torch
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(weights, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self.bufs = {}
        for n in names:
            shape = tuple(self.engine.get_tensor_shape(n))
            self.bufs[n] = torch.zeros(shape, dtype=torch.float32,
                                       device="cuda")
            self.ctx.set_tensor_address(n, self.bufs[n].data_ptr())
        self.in_name = names[0]
        self.da_name = names[1]             # export order: img -> (da, ll)

    def __call__(self, img_bgr):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        ratio = min(self.IN_H / h, self.IN_W / w)
        nh, nw = int(round(h * ratio)), int(round(w * ratio))
        top = (self.IN_H - nh) // 2
        left = (self.IN_W - nw) // 2
        canvas = np.full((self.IN_H, self.IN_W, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(
            img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(canvas[:, :, ::-1].copy()).cuda()
        t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        self.bufs[self.in_name].copy_(t)
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        da = self.bufs[self.da_name][:, :, top:top + nh, left:left + nw]
        da = torch.nn.functional.interpolate(da, size=(h, w),
                                             mode="bilinear")
        return (torch.argmax(da, dim=1).squeeze(0).cpu().numpy() == 1)


def create_segmenter(method="hsv", **kw):
    """Factory: 'hsv' (classical, no deps) or 'twinlitenet' (learned,
    needs torch + cloned repo + weights -- kw: repo_path, weights, config)."""
    if method == "hsv":
        return HsvSegmenter(**kw)
    if method == "twinlitenet":
        if str(kw.get("weights", "")).endswith(".engine"):
            return TwinLiteTRTSegmenter(**kw)
        return TwinLiteSegmenter(**kw)
    raise ValueError("unknown segmentation method: %r" % (method,))


def segment_road(img_bgr, method: str = "hsv", **kw) -> np.ndarray:
    """
    Dispatch to a road-segmentation backend via the factory. 'hsv' is the
    classical default; 'twinlitenet' is the learned model (needs torch +
    a cloned TwinLiteNetPlus repo + weights -- see create_segmenter).
    """
    return create_segmenter(method, **kw)(img_bgr)


def white_line_mask(img_bgr, min_grass_frac=0.10, min_elong=3.0):
    """Painted white course lines (IGVC: chalk/paint on grass) as a boolean
    mask -- used as a NEGATIVE on the road mask so lines become off-road
    boundaries in the costmap. Classical and cheap (~2 ms): white gate
    restricted to grass-adjacent pixels, elongated components only.
    Returns all-False when the scene has too little grass (indoors, roads).
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    if grass.mean() < min_grass_frac * 255:
        return np.zeros((h, w), bool)
    white = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    white &= cv2.dilate(grass, np.ones((25, 25), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
    out = np.zeros((h, w), bool)
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 0.0006 * h * w:
            continue
        (rw, rh) = cv2.minAreaRect(c)[1]
        if min(rw, rh) == 0 or max(rw, rh) / max(min(rw, rh), 1.0) < min_elong:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [c.reshape(-1, 2)], 255)
        out |= m > 0
    return out
