# driving_seg — mission prompt

Use this prompt to (re)start any session or agent working on this project.

---

Build and maintain **driving_seg**: a real-time, multi-model semantic
highlighting stack for an autonomous ground vehicle. It paints translucent
colored AREAS over everything that matters for driving — never bounding
boxes — and exposes the same masks as machine-readable layers.

## Classes and colors (fixed contract)

| class | color (BGR) | source model |
|---|---|---|
| person | red (60,60,230) | scene |
| vehicle (car/truck/bus/motorcycle/bicycle) | blue (230,120,60) | scene |
| traffic_sign (stop sign) | yellow (60,220,230) | scene |
| traffic_light | amber (60,170,255) | scene |
| road (drivable area) | green (80,200,80) | road |
| lane_line (painted road lanes) | cyan (230,230,80) | road |
| white_line (course lines, e.g. painted on grass) | white (255,255,255) | course |
| cone (traffic cones) | orange (40,120,255) | course |

Priority when classes overlap a pixel (highest wins):
person > cone > vehicle > traffic_sign > traffic_light > white_line >
lane_line > road.

## Architecture (three specialized models + fusion)

- `scene`: nano instance-segmentation model, COCO-pretrained (ultralytics),
  masks only — boxes are never drawn.
- `road`: dual-head drivable-area + lane-line segmenter (YOLOPv2-class).
- `course`: fine-tuned nano seg model for {cone, white_line} — the two
  classes nothing pretrained covers. Trained in-repo (tools/train_course.py).
- `fusion`: composites per-class boolean masks into one overlay frame
  (alpha ~0.45, thin brightened contour per region) and returns the raw
  mask dict {class_name: HxW bool}.

Every model wrapper implements: `predict(bgr) -> dict[str, np.ndarray bool]`
and is import-lazy (the package and its tests must work with zero model
deps installed).

## Hard constraints

- Jetson AGX Orin (JetPack 6.1, TensorRT 10.3, FP16) is the deployment
  target: model choices must have a TRT export path and nano-class latency
  (target: full 3-model pipeline >= 10 Hz on Orin at 640x384; >= 60 FPS on
  the RTX 5090 dev box).
- Python 3.8-compatible syntax (Jetson floor).
- This repo is standalone — it must NOT import from or modify
  carla-nav2-avl, IGVC, or any other workspace.
- Dev box: RTX 5090 32 GB (torch 2.9 cu128). Jetson wheels come from
  https://pypi.jetson-ai-lab.io/jp6/cu126 (torch 2.8 / numpy<2 there).

## Entry points

- `python -m driving_seg.demo --source <img|dir|video|cam> --out out/` —
  highlighted media + optional per-class mask dump + per-stage ms table.
- `tools/train_course.py` — fine-tune the course model from datasets/.
- `tools/export_trt.py` — FP16 TRT engines for all three models (run ON the
  Jetson for deployment engines).
- `pytest test -q` — offline suite (synthetic masks, no weights needed).

## Definition of done for any change

Offline tests pass; demo runs end-to-end on testdata/ producing overlays
where road is green, cones orange, white lines white, etc.; per-stage
timings printed; no bounding boxes visible anywhere.
