# carla-nav2-avl — feature/alexander

Sim-to-real perception for our autonomous ground vehicle: validate in CARLA
with the same 3-camera layout as the car, deploy the identical ROS2 stack to
the car computer ("dinosaur", Jetson AGX Orin, ROS2 Humble). CARLA is
x86-only and never runs on the Jetson — only the sensor source changes.

> New here? Read `CLAUDE.md` next — it says exactly what is real, what is
> stub, and the conventions that bind changes.
> Finishing the project? **`FINALIZE.md`** is the ordered master plan
> (phases, commands, acceptance criteria) to take this to competition-ready.

## What's on this branch (all real, tested)

| where | what | start here |
|---|---|---|
| `ros2_ws/src/perception_costmap/` | camera+lidar → Nav2 costmap. Multi-camera BEV fusion, TwinLiteNet road seg, YOLOv8-TensorRT obstacles, temporal filter. 39 offline tests. | its `README.md`, then `DESIGN.md` |
| `ros2_ws/src/perception_costmap/deploy/` | on-car ops: full-stack bringup (tmux), GMSL camera recovery, boot-time systemd unit, live phone dashboard | `deploy/README.md` |
| `driving_seg/` | **NEW: multi-model driving segmentation — area highlighting, no bounding boxes.** People, vehicles, signs, lights, road, lanes, **cones**, white lines. 74 FPS on an RTX 5090; TensorRT path for the Jetson. | `driving_seg/README.md` and **`driving_seg/docs/CONE_DETECTION.md`** |
| `perception/` | Adam Castillo's original prototype scripts the package grew from (credits preserved) | — |

Not real yet (don't build on): `collision_guard`, `route_planner`,
`sdc_common`, `controller`, `sdc_bringup` — stubs, see `CLAUDE.md`.

## Cone detection — the 60-second version

We need cones (and painted white course lines) segmented as *areas*, and no
pretrained model knows them. The approach lives in `driving_seg/`:

1. `driving_seg/` runs three specialized models in parallel and fuses them:
   COCO nano-seg (people/vehicles/signs), YOLOPv2 (road/lanes), and a
   **fine-tuned cone+line model** we train ourselves.
2. Training data is built by *self-distillation*: scrape freely-licensed
   photos, auto-label cones by color+geometry, multiply them with copy-paste
   augmentation, then fine-tune a nano seg net that generalizes past the
   color heuristic. Full story + how to retrain with photos of OUR cones:
   **`driving_seg/docs/CONE_DETECTION.md`**.
3. The trained model is committed (`driving_seg/models/course.pt`) — clone
   and run, no training required.

## Quick starts

Perception costmap (no ROS needed for tests):

    cd ros2_ws/src/perception_costmap
    PYTHONPATH=. python3 -m pytest test -q          # 39 passed

Driving segmentation demo (any machine with a GPU):

    cd driving_seg
    pip install ultralytics
    curl -L -o models/yolopv2.pt https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
    PYTHONPATH=. python3 -m driving_seg.demo --source <image-or-video> --out out/

On the car (dinosaur): the whole stack auto-starts on boot
(`percept-stack.service`); manual restart via
`ros2_ws/src/perception_costmap/deploy/full_stack_restart.sh`;
live view at `http://<car-ip>:8090`, joystick at `https://<car-ip>:8000`.

## Team

alexander (arassal) leads this branch; jchy05, AdamCastillo07, Ad-Tap have
their own feature branches. Commit style and conventions: see `CLAUDE.md`.
Development workflow: `CONTRIBUTION_GUIDE.md`.
