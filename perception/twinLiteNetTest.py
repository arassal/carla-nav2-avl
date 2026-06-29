"""
twinlitenet_dashcam.py
-----------------------
Runs TwinLiteNet+ (drivable-area + lane segmentation) on a dashcam video.

This is a CPU/GPU-flexible wrapper around the official TwinLiteNetPlus repo's
model code. The repo's own demo.py *hardcodes* CUDA + FP16, which won't run
on a laptop without a real GPU -- this script auto-detects CPU vs GPU instead,
and replaces their custom dataset-loading classes with a plain OpenCV video
loop, which is easier to follow and modify.

-----------------------------------------------------------------------------
ONE-TIME SETUP (do this before running this script):
-----------------------------------------------------------------------------
1. Clone the official model repo (this script imports its model definition,
   it does not redefine the network architecture itself):

       git clone https://github.com/chequanghuy/TwinLiteNetPlus.git

2. Install dependencies (a CPU-only torch install is fine on the laptop):

       pip install torch torchvision opencv-python pyyaml tqdm

3. Download pretrained weights from the link in that repo's README
   (Google Drive folder, look for nano.pth / small.pth / medium.pth / large.pth).
   "nano" is recommended for a Ryzen 3 laptop with no discrete GPU --
   it's a 34K-parameter model, by far the lightest of the four.

4. Place this script EITHER:
     a) inside the cloned TwinLiteNetPlus folder, or
     b) anywhere, and pass --repo-path pointing at the cloned folder.

-----------------------------------------------------------------------------
RUN:
-----------------------------------------------------------------------------
    python3 twinlitenet_dashcam.py \
        --repo-path ./TwinLiteNetPlus \
        --weight ./TwinLiteNetPlus/pretrained/nano.pth \
        --config nano \
        --video my_dashcam_footage.mp4 \
        --out result.mp4

-----------------------------------------------------------------------------
NOTE -- I have not been able to test this end-to-end myself: it requires the
cloned repo, downloaded weights, and a GPU/CPU torch install, none of which
are available in my sandbox. If something errors when you run it, paste the
traceback back to me and we'll fix it together rather than guessing blind.
-----------------------------------------------------------------------------
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# ----------------------------------------------------------------------
# PREPROCESSING -- "letterbox" resize (same idea YOLO-family models use):
# resize to fit inside img_size x img_size while keeping aspect ratio,
# then pad the rest with gray so the image isn't distorted.
# ----------------------------------------------------------------------
def letterbox(img, new_size=640, pad_color=(114, 114, 114)):
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


# ----------------------------------------------------------------------
# VISUALIZATION -- green = drivable area, red = lane line
# (matches the color convention used in the official repo's demo.py)
# ----------------------------------------------------------------------
def overlay_masks(frame_bgr, da_mask, ll_mask, alpha=0.5):
    overlay = np.zeros_like(frame_bgr)
    overlay[da_mask == 1] = (0, 255, 0)   # green: drivable area
    overlay[ll_mask == 1] = (0, 0, 255)   # red: lane line

    has_overlay = (overlay.sum(axis=2) != 0)
    out = frame_bgr.copy()
    out[has_overlay] = (out[has_overlay] * (1 - alpha) + overlay[has_overlay] * alpha).astype(np.uint8)
    return out


def load_model(repo_path, weight_path, config, device):
    sys.path.insert(0, str(repo_path))
    from model.model import TwinLiteNetPlus  # provided by the cloned repo

    model_args = argparse.Namespace(config=config)
    model = TwinLiteNetPlus(model_args)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_on_video(model, video_path, out_path, img_size, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    t0 = time.time()

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            padded, ratio, (pad_left, pad_top) = letterbox(frame, img_size)

            tensor = torch.from_numpy(padded).to(device).float()
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # HWC -> NCHW, normalize

            da_out, ll_out = model(tensor)

            if frame_count == 1:
                print(f"DEBUG: input tensor shape: {tensor.shape}")
                print(f"DEBUG: da_out shape: {da_out.shape}")
                print(f"DEBUG: ll_out shape: {ll_out.shape}")
                print(f"DEBUG: img_size={img_size}, pad_left={pad_left}, pad_top={pad_top}, ratio={ratio}")

            # undo the letterbox padding/scaling so masks line up with
            # the original (un-padded, un-resized) frame
            da_crop = da_out[:, :, pad_top:img_size - pad_top, pad_left:img_size - pad_left]
            ll_crop = ll_out[:, :, pad_top:img_size - pad_top, pad_left:img_size - pad_left]

            da_resized = torch.nn.functional.interpolate(da_crop, size=(h, w), mode="bilinear")
            ll_resized = torch.nn.functional.interpolate(ll_crop, size=(h, w), mode="bilinear")

            da_mask = torch.argmax(da_resized, dim=1).squeeze().cpu().numpy()
            ll_mask = torch.argmax(ll_resized, dim=1).squeeze().cpu().numpy()

            out_frame = overlay_masks(frame, da_mask, ll_mask)
            writer.write(out_frame)

            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                print(f"Processed {frame_count} frames ({frame_count / elapsed:.1f} FPS so far)")

    cap.release()
    writer.release()
    print(f"Done. Saved to {out_path} ({frame_count} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=str, required=True,
                         help="path to the cloned TwinLiteNetPlus repo (so model/model.py can be imported)")
    parser.add_argument("--weight", type=str, required=True, help="path to pretrained .pth weights")
    parser.add_argument("--config", type=str, default="nano",
                         choices=["nano", "small", "medium", "large"],
                         help="model size -- 'nano' recommended for CPU/laptop use")
    parser.add_argument("--video", type=str, required=True, help="path to dashcam video")
    parser.add_argument("--out", type=str, default="result.mp4")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default=None,
                         help="'cpu' or 'cuda'. If omitted, auto-detects GPU availability.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(Path(args.repo_path), args.weight, args.config, device)
    run_on_video(model, args.video, args.out, args.img_size, device)


if __name__ == "__main__":
    main()