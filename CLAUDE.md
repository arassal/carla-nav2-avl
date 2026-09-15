# Notes for AI assistants

Humans and AI follow the same rules. Read `README.md` for what the project is
and what's real vs. stub; the conventions that bind every change are in
`CONTRIBUTING.md`, imported here:

@CONTRIBUTING.md

This file adds only what an AI session is likely to get wrong.

## Don't trust a green test run

The offline suite covers the ROS-free modules only. `costmap_node.py` — the
main node — currently **cannot be imported**: it needs `detection_schedule`,
`inference_worker`, `sample_buffer` and `health`, which were never committed
(`ISSUES.md` B1). Never report "tests pass" as "the stack works".

## Missing files: ask, don't reimplement

Eleven files referenced by the code or docs are in no commit (list in
`ISSUES.md` D2). They most likely exist uncommitted on the car and have been
validated there. Do not write replacements without asking Alexander —
a from-scratch version can silently discard working, on-car-tested code.

## History that looks like a mistake but isn't

- `perception/` (Adam Castillo's prototype) was deleted in `060141f`. The
  docstrings in `segmentation.py` and `obstacles.py` that cite it as their
  source are accurate attribution — leave them. Credit also lives in
  `CONTRIBUTORS.md` and `.mailmap`.
- The original implementation plan was deleted in `e0cf788`; read it with
  `git show e0cf788^:docs/plans/2026-07-01-perception-v2-sim-to-real.md`.
- `models/cone_det.pt` and `driving_seg/models/cone_det.pt` are identical on
  purpose, so each package is self-contained.
- `config/nav2_costmap_params.yaml` is reference-only (see `DEPLOY.md`); it
  reads `/perception/obstacle_points`, not the costmap cloud.

## What needs a human or hardware

Don't claim these are done from a laptop session:

- Per-camera IPM calibration — YAML `ipm_*` values are placeholders; use
  `tools/ipm_overlay.py` against a real frame.
- CARLA smoke test and IoU table — x86 sim box with CARLA 0.9.16.
- Anything touching Nav2, localization or driving — needs the car or the
  `avros_*` packages from github.com/Paarseus/IGVC_ROS2.
- TensorRT export and benchmarks — must run on the Jetson itself.

## Team

alexander (arassal) leads; jchy05, AdamCastillo07 and Ad-Tap are mentees with
their own feature branches. Machine access details are not in this repo.
