# driving_seg design — 2026-07-03

## Goal

Real-time area highlighting (no bounding boxes) of the eight driving
classes in PROMPT.md, from three specialized models fused into one overlay
+ per-class masks. Jetson-first sizing, developed on the RTX 5090.

## Decisions (approved 2026-07-03)

1. **Cones + white lines**: fine-tune a real nano seg model on public
   datasets (5090 makes this ~1-2 h). Classical CV only as an emergency
   fallback if datasets prove unusable.
2. **Jetson-first**: nano-class models, 640x384-ish inputs, TRT FP16 path
   mandatory for every model picked.
3. **v1 output**: standalone CLI demo; ROS adapter is a later thin layer.

## Components

```
driving_seg/
  config.py      # class registry: names, colors, priority, model routing
  fusion.py      # masks dict -> overlay frame (pure numpy/cv2, testable)
  models/
    base.py      # SegModel protocol: predict(bgr) -> {class: bool mask}
    scene.py     # ultralytics nano-seg wrapper (COCO subset -> our classes)
    road.py      # YOLOPv2-class wrapper (drivable + lane heads)
    course.py    # fine-tuned wrapper (cone, white_line)
  pipeline.py    # runs models (thread-parallel), merges dicts, times stages
  demo.py        # CLI: image/dir/video/cam -> overlay media + timings
tools/
  train_course.py    # fine-tune course model from datasets/
  export_trt.py      # FP16 engines (run on target hardware)
test/                # offline: fusion, config, wrapper contracts (mocked)
testdata/            # agent-gathered street/cones/grass-lines/video media
docs/specs/          # this file
PROMPT.md            # canonical mission prompt
PLAN.md              # execution plan + status
```

## Data flow

frame (BGR) -> [scene, road, course] in parallel threads (each lazy-loads
its backend on first use; missing weights = empty contribution + warning)
-> per-class bool masks resized to frame -> fusion.compose(masks, frame)
-> overlay + mask dict. Demo writes overlay media; pipeline returns
timings per stage.

## Error handling

- A model that fails to load logs once and contributes nothing (pipeline
  degrades gracefully — never crashes the demo).
- Masks are validated to shape/bool at the wrapper boundary.
- Video writer falls back to per-frame PNGs if codec unavailable.

## Testing

- Offline (no weights): fusion priority/alpha/contour math on synthetic
  masks; config integrity (every class routed, colors unique, priority
  total); pipeline with stub models; asymmetric-fixture orientation checks
  on overlay placement (lesson from the BEV mirror bug).
- With weights: golden smoke — demo over testdata/ produces nonzero road
  mask on street scenes, nonzero cone mask on cone photos.
- Bench: per-stage ms + end-to-end FPS table (5090 now, Orin later).

## Risks

- Dataset quality for white-lines-on-grass is the weakest link; mitigation:
  research agent ranks options first, sports-field chalk lines transfer
  visually, and classical white-extraction fallback remains possible
  inside course.py without changing the interface.
- Ultralytics on torch 2.9/cu128: if a version pin fights the 5090, pin
  ultralytics latest + let it resolve; worst case dev in fp32 pytorch and
  TRT only on Jetson.
