# Notes for AI assistants

Humans and AI follow the same rules. Read `README.md` for what the project is
and what's real vs. stub; the conventions that bind every change are in
`CONTRIBUTING.md`, imported here:

@CONTRIBUTING.md

This file adds only what an AI session is likely to get wrong.

## Don't trust a green test run

The offline suite covers the ROS-free modules only — no test starts
`costmap_node.py`. Never report "tests pass" as "the stack works". To check
the node itself, run it against a synthetic camera the way
`logs/results/2026-09-14_costmap-node-live-run.md` (PR #3) does. That run found two
bugs the suite could not, one of which only appears under Ubuntu's OpenCV 4.6,
not pip's 4.11 (`ISSUES.md` C5, C6).

## "Missing" files: fetch and ask before writing

In September 2026 eleven files the code imported were absent from `copy`
(`ISSUES.md` B1-B3, D2). They were not unwritten: they sat untracked on
Alexander's machine and were committed in `0fedca3`. A from-scratch
replacement written in the meantime had to be thrown away. If something
looks missing, `git fetch` first, then ask the team before implementing it.

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
