# Models

Portable weights are committed here (`.pt` / `.pth`). The TensorRT `.engine`
files are **not** committed — they are compiled for a specific GPU + CUDA +
TensorRT version and must be rebuilt on the target machine:

```bash
# on the target Jetson, after installing deps (see ../DEPLOY.md §1):
cd ros2_ws/src/perception_costmap
python3 tools/export_trt.py --weights ../../../models/yolov8n.pt   # -> yolov8n.engine
python3 tools/export_trt.py --weights ../../../models/cone_det.pt  # -> cone_det.engine
# TwinLiteNet: see DEPLOY.md; nano.pth is the pytorch fallback if no .engine yet
```

Point the perception config at this directory with:

```bash
export AVL_MODELS_DIR=/absolute/path/to/repo/models
```

`config/perception_dinosaur.yaml` references models as `${AVL_MODELS_DIR}/...`.

## Why there are two `models/` directories
- `/models/` (this dir) feeds the **ROS2 `perception_costmap`** package on the car.
- `/driving_seg/models/` feeds the standalone **`driving_seg`** segmentation package.

`cone_det.pt` is intentionally present in both so each package is
self-contained; they are the same weights.
