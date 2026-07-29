"""CLI demo: image / directory / video / webcam -> highlighted media.

    python -m driving_seg.demo --source testdata/street --out out/
    python -m driving_seg.demo --source clip.mp4 --out out/
    python -m driving_seg.demo --source 0 --out out/          # webcam
"""

import argparse
import os
import sys
import time

import cv2

from .pipeline import Pipeline

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".webm", ".mkv")


def _timing_line(p):
    return "  ".join("%s %.1fms" % (k, v) for k, v in sorted(p.timings.items()))


def run_images(pipe, paths, out_dir, dump_masks):
    import numpy as np
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print("skip (unreadable):", path)
            continue
        overlay, masks = pipe.overlay(img)
        base = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(out_dir, base + "_seg.jpg")
        cv2.imwrite(dst, overlay)
        if dump_masks:
            np.savez_compressed(os.path.join(out_dir, base + "_masks.npz"),
                                **{k: v for k, v in masks.items()})
        print("%s -> %s   [%s]" % (os.path.basename(path), dst, _timing_line(pipe)))


def run_video(pipe, source, out_dir, max_frames=None):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit("cannot open source: %s" % source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = "camera" if source.isdigit() else os.path.splitext(os.path.basename(source))[0]
    dst = os.path.join(out_dir, name + "_seg.mp4")
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        overlay, _ = pipe.overlay(frame)
        vw.write(overlay)
        n += 1
        if n % 30 == 0:
            print("frame %d  [%s]" % (n, _timing_line(pipe)))
    cap.release(); vw.release()
    dt = time.perf_counter() - t0
    print("%d frames -> %s  (%.1f FPS end-to-end)" % (n, dst, n / dt if dt else 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="image, directory, video file, or camera index")
    ap.add_argument("--out", default="out")
    ap.add_argument("--masks", action="store_true",
                    help="also dump per-class masks (.npz) for images")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scene-weights", default="models/yolo11n-seg.pt")
    ap.add_argument("--road-weights", default="models/yolopv2.pt")
    ap.add_argument("--course-weights", default="models/course.pt")
    args = ap.parse_args()

    from .models.scene import SceneModel
    from .models.road import RoadModel
    from .models.course import CourseModel
    pipe = Pipeline([SceneModel(args.scene_weights),
                     RoadModel(args.road_weights),
                     CourseModel(args.course_weights)])

    os.makedirs(args.out, exist_ok=True)
    src = args.source
    if os.path.isdir(src):
        paths = sorted(os.path.join(src, f) for f in os.listdir(src)
                       if f.lower().endswith(IMG_EXT))
        run_images(pipe, paths, args.out, args.masks)
    elif src.lower().endswith(IMG_EXT):
        run_images(pipe, [src], args.out, args.masks)
    else:
        run_video(pipe, src, args.out, args.max_frames)


if __name__ == "__main__":
    main()
