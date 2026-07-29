"""Per-stage latency bench.

    python3 tools/bench.py --frames 50 [--image path]
"""

import argparse
import statistics
import sys

import cv2

sys.path.insert(0, ".")
from driving_seg.pipeline import Pipeline                      # noqa: E402
from driving_seg.models.scene import SceneModel                # noqa: E402
from driving_seg.models.road import RoadModel                  # noqa: E402
from driving_seg.models.course import CourseModel              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--image", default="testdata/street/000000174482.jpg")
    ap.add_argument("--scene-weights", default="models/yolo11n-seg.pt")
    ap.add_argument("--road-weights", default="models/yolopv2.pt")
    ap.add_argument("--course-weights", default="models/course.pt")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit("cannot read " + args.image)
    pipe = Pipeline([SceneModel(args.scene_weights),
                     RoadModel(args.road_weights),
                     CourseModel(args.course_weights)])
    pipe.overlay(img)                                  # warmup
    hist = {}
    for _ in range(args.frames):
        pipe.overlay(img)
        for k, v in pipe.timings.items():
            hist.setdefault(k, []).append(v)
    print("stage        median ms   p95 ms")
    for k in sorted(hist):
        xs = sorted(hist[k])
        print("%-12s %8.1f %8.1f" % (k, statistics.median(xs),
                                     xs[int(0.95 * len(xs)) - 1]))
    med = statistics.median(hist["total"])
    print("end-to-end ~%.1f FPS (parallel stages)" % (1000.0 / med))


if __name__ == "__main__":
    main()
