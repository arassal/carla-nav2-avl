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
                     val_hi: int = 200,
                     use_clahe: bool = True,
                     min_blob_frac: float = 0.15) -> np.ndarray:
    """
    Classical road mask: asphalt is low-saturation, mid-brightness. Threshold
    in HSV, clean up, then keep connected blobs, and returns a boolean mask
    the size of the image.

    2026-08-12: two shadow-robustness fixes, made after a real stuck-in-shadow
    incident (vehicle treated shadowed pavement ahead as off-road and refused
    to proceed):

    1. CLAHE (local contrast normalization) on the L channel before HSV
       thresholding. A raw global val_lo/val_hi cutoff can't tell "this pixel
       is genuinely dark asphalt/gravel" from "this pixel is normal asphalt
       sitting in a shadow" -- both just read as low V. CLAHE equalizes
       brightness using LOCAL neighborhood statistics, so the same physical
       surface reads similarly whether lit or shadowed, instead of a shadow
       band pushing V below val_lo and flipping pixels to non-road. CLAHE
       only touches lightness, not hue/saturation, so it doesn't loosen the
       max_sat gate that keeps green grass correctly classified as off-road
       (see perception_dinosaur.yaml offroad_cost:97 / costmap_to_cloud.py --
       this segmenter feeds that same road/grass distinction).
    2. Keep every blob at least min_blob_frac the size of the largest, not
       ONLY the single largest. A shadow band crossing the road can visually
       split one contiguous road region into two disconnected blobs even
       after CLAHE; picking only the biggest one would silently drop the far
       side of the shadow as "not road" -- exactly the stuck-at-shadow
       symptom, just from the blob-selection step instead of the threshold.

    HSV ranges are still lighting-dependent at the margins -- recalibrate for
    the real camera, or switch to the learned segmenter, if this isn't enough.
    """
    if use_clahe:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s < max_sat) & (v > val_lo) & (v < val_hi)).astype(np.uint8) * 255

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = areas.max()
        keep_labels = 1 + np.where(areas >= min_blob_frac * largest_area)[0]
        # Label ids are dense integers in [0, num), so membership is a direct
        # table lookup. np.isin sorts and searches per pixel instead, which
        # profiled as ~10% of this node's main-thread time on the Jetson.
        # Same result, O(pixels) instead of O(pixels log keep).
        keep_lut = np.zeros(num, dtype=bool)
        keep_lut[keep_labels] = True
        mask = keep_lut[labels].astype(np.uint8) * 255
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


class NullSegmenter:
    """Everything observed is drivable.

    For a course marked out with painted lines on grass there is no "road" to
    find -- the drivable area is simply everywhere the cameras can see that is
    not a line. Road segmentation there is worse than useless: HSV hunts for
    asphalt, finds none, and the whole field reads as off-road. Pair this with
    white_line_mode: obstacle so the lines are the only thing that costs.
    """

    def __init__(self, **_ignored):
        pass

    def __call__(self, img_bgr):
        return np.ones(img_bgr.shape[:2], dtype=bool)


def create_segmenter(method="hsv", **kw):
    """Factory: 'hsv' (classical, no deps), 'none' (everything drivable --
    line-marked courses), or 'twinlitenet' (learned, needs torch + cloned
    repo + weights -- kw: repo_path, weights, config)."""
    if method == "none":
        return NullSegmenter(**kw)
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


def white_feature_masks(img_bgr, min_elong=3.0, require_grass=True,
                        min_grass_frac=0.10, white_v_min=165, white_s_max=70,
                        roi_top_frac=0.0, min_circularity=0.70,
                        min_area_frac=0.0006, max_pothole_area_frac=0.15,
                        pothole_max_elong=1.8):
    """Split bright-white blobs into course LINES and simulated POTHOLES.

    IGVC 2026 runs AutoNav on ASPHALT, with white boundary lines ~3 in
    (7.6 cm) wide TAPED on the pavement -- not painted on grass. ("grass"
    does not appear anywhere in the 2026 rulebook; see rules S II.2.)

    Two different white things sit on that course and must not be confused:

      * boundary lines      -- long and thin (elongation >= min_elong)
      * simulated potholes  -- 2 ft (0.61 m) SOLID WHITE CIRCLES

    Potholes must be avoided or the run ends, and a circle's elongation is
    ~1.0, so the very filter that makes the line detector precise rejects
    every pothole by construction. They are therefore classified separately
    here rather than being silently dropped.

    Returns ``(lines, potholes)`` as boolean masks.

    CAVEAT: these shape gates are still applied in PERSPECTIVE space, where a
    fixed-width line spans many pixels near the vehicle and under one pixel
    far away -- so both the area and elongation gates are range-dependent.
    Moving this into the BEV, where 7.6 cm is a constant pixel width, is
    Phase 1 of the perception plan; until then these are near-field values.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, white_v_min), (180, white_s_max, 255))
    if roi_top_frac > 0.0:
        # Everything this detector finds is fed through IPM, which assumes the
        # pixel lies on the ground plane. Anything ABOVE the ground -- ceiling
        # lights, windows, a bright wall -- violates that and back-projects to
        # an enormous far-field patch. Measured 2026-09-17 indoors: 558 bright
        # pixels, all in the top 30% of the frame, became 16% of the grid
        # LETHAL while the actual floor line was never detected.
        white[:int(h * roi_top_frac), :] = 0
    if require_grass:
        # Only meaningful at a grass venue. On the IGVC asphalt course there
        # is no green at all, so competition presets set require_grass False
        # and rely on the shape gates below.
        grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
        if grass.mean() < min_grass_frac * 255:
            return np.zeros((h, w), bool), np.zeros((h, w), bool)
        white &= cv2.dilate(grass, np.ones((25, 25), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))

    lines = np.zeros((h, w), bool)
    potholes = np.zeros((h, w), bool)
    frame_area = float(h * w)
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_frac * frame_area:
            continue
        (rw, rh) = cv2.minAreaRect(c)[1]
        elong = max(rw, rh) / max(min(rw, rh), 1.0)
        perim = cv2.arcLength(c, True)
        circularity = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0

        if (circularity >= min_circularity
                and elong <= pothole_max_elong
                and area <= max_pothole_area_frac * frame_area):
            target = potholes
        elif elong >= min_elong:
            target = lines
        else:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [c.reshape(-1, 2)], 255)
        target |= m > 0
    return lines, potholes


def white_line_mask(img_bgr, min_grass_frac=0.10, min_elong=3.0,
                    require_grass=True, white_v_min=165, white_s_max=70,
                    roi_top_frac=0.0):
    """Course lines only -- compatibility wrapper over white_feature_masks().

    NOTE: this DISCARDS the pothole mask. A caller that must not drive over a
    simulated pothole has to use white_feature_masks() and consume both.
    """
    lines, _ = white_feature_masks(
        img_bgr, min_elong=min_elong, require_grass=require_grass,
        min_grass_frac=min_grass_frac, white_v_min=white_v_min,
        white_s_max=white_s_max, roi_top_frac=roi_top_frac)
    return lines
