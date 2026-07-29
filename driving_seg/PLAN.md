# Execution plan (status)

1. [x] Folder + git + PROMPT.md + design spec
2. [x] Core package (config/fusion/wrappers/pipeline/demo) — 7 offline tests
3. [x] Weights + demo verified on street scenes
4. [x] Course model: v1 (93 scraped imgs) -> v2 (103 real + 299 copy-paste
       synthetic). Val mask AP50: cone 0.295, white_line 0.431 (v1: 0.293 /
       0.006). Solid on orange cones; striped/white-banded cones still weak —
       fix is photographing OUR cones (docs/CONE_DETECTION.md).
5. [x] Bench (RTX 5090, 40 frames, parallel stages): scene 9.0 ms, road
       7.6 ms, course 7.5 ms median -> total 13.6 ms ≈ 74 FPS.
       TRT export: tools/export_trt.py (run on the Jetson).
6. [x] Committed to carla-nav2-avl feature/alexander under driving_seg/.

Open: white_line real-world data, striped-cone recall, Jetson TRT bench.
