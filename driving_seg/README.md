# driving_seg

Real-time multi-model **area highlighting** for driving — no bounding
boxes, ever. Three specialized models fused into one overlay + per-class
masks. See PROMPT.md for the full contract (classes, colors, priorities,
constraints) and PLAN.md for status.

## Quick start

    pip install ultralytics
    bash tools/fetch_testdata.sh            # sample media (optional)
    python3 -m driving_seg.demo --source testdata/street --out out/

Weights land in `models/`: `yolo11n-seg.pt` (auto-downloaded by
ultralytics), `yolopv2.pt` (CAIC-AD/YOLOPv2 release), `course.pt`
(train it: `tools/build_course_dataset.py` then `tools/train_course.py`).
A missing model degrades gracefully — its classes just don't appear.

## Layout

    driving_seg/     package: config, fusion, models/, pipeline, demo
    tools/           dataset builder, trainer, TRT export, bench
    test/            offline suite: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
                       PYTHONPATH=. python3 -m pytest test -q
    docs/specs/      design doc
    testdata/, datasets/, models/, out/   (gitignored artifacts)

## Deployment (Jetson)

Build engines ON the Jetson: `tools/export_trt.py` (+ trtexec for
YOLOPv2). Jetson wheels: https://pypi.jetson-ai-lab.io/jp6/cu126,
keep numpy<2 there.
