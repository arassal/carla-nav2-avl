# Weights

- `course.pt` (committed): fine-tuned YOLO11n-seg for {cone, white_line} v2.
- `cone_det.pt` (committed, 22 MB): dedicated cone detector — ExStella/Traffic-cones
  (Hugging Face, Apache-2.0, YOLOv8s). Catches striped/white-banded cones. Source:
  https://huggingface.co/ExStella/Traffic-cones (resolve/main/best.pt).
  Used by models/course.py to seed cone AREA masks; boxes are never drawn.
- `yolo11n-seg.pt`: auto-downloaded by ultralytics on first run.
- `yolopv2.pt` (156 MB, not committed):
  curl -L -o models/yolopv2.pt https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
