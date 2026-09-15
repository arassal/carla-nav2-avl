# Lean launch on dinosaur — first attempt (partial)

**Date:** 2026-09-15 · **Car:** dinosaur (Jetson AGX Orin, Humble, MAXN, ZED SDK 5.2.0)
**Code:** `~/chris_test/carla-nav2-avl`, branch `test/all-prs` (`copy` @ `0fedca3` + PRs #1-#4)

**Result: incomplete.** The launch behaved as designed and the ZED drivers got
the lean settings, but the cameras could not be opened because of a hardware
state left by a power cycle. The CPU/GPU comparison still needs a run after a
Jetson reboot.

## 1. Baseline: the boot stack (`percept-stack.service`), all 3 cameras

Taken 13:05 after `clean_camera_restart.sh`, with the boot costmap restored on
domain 0. 30 s sample; per-process CPU from `/proc` over the window
(100% = one core).

| | boot stack |
|---|---|
| `/perception/costmap` | **10.13 Hz** |
| RGB front / left / right | 7.9 / 14.9 / 15.0 Hz |
| depth front | 7.9 Hz |
| ZED topics per camera | 27 (incl. `point_cloud`, `odom`/`pose`) |
| CPU, mean of 12 cores | **34.8%** · load avg 8.6 |
| GPU (GR3D) | **33.9%** mean, 99% peak |
| RAM | 7.6 GB |
| top processes | costmap_node 65%, **rviz2 44%**, zed_left 31%, **viz_node 21%**, zed_front 15%, costmap_rgb 8%, xsens 5%, ekf 4% |

zed_right was streaming but didn't appear in the per-process sample; its
container was probably relaunched mid-window.

## 2. Lean launch

Between the baseline and this run, someone power-cycled hardware on the car
(not the Jetson, which stayed up). zed_right stopped streaming. Then:

    sudo systemctl stop percept-stack; tmux -L percept kill-session -t percept
    sudo systemctl restart nvargus-daemon; sudo systemctl restart zed_x_daemon
    ros2 launch perception_costmap perception_stack.launch.py

**What worked:**

    [perception_stack] domain=0 cameras=['front', 'left', 'right'] already running=[] start sensors=True costmap=True
    [perception_stack] zed_front starts in 15 s / zed_left in 55 s / zed_right in 95 s

Sensors (robot_state_publisher, Velodyne, Xsens) and the costmap node started
immediately; the YOLO and cone TensorRT engines loaded. The front ZED driver
loaded `zed_perception_front.yaml` and echoed exactly the intended settings:

    Publish RGB image: TRUE          Publish Point Cloud: FALSE
    Publish Depth Map: TRUE          Publish Odometry/Pose: FALSE
    Publish Depth Confidence: TRUE   Publish IMU: FALSE
    Positional tracking enabled: FALSE
    Grab Compute Capping FPS: 8      Publish framerate [Hz]: 8
    Depth Stabilization: 0           Depth mode: NEURAL LIGHT
    Camera SN: 42569280              Camera resolution: SVGA

**What failed:** at `=== CAMERA OPENING ===` the front container (13:12:18) and
the left container (13:12:58) each hit

    (Argus) Error BadParameter: (propagating from src/rpc/socket/client/CustomMethods.cpp, function createBuffer())
    process has died [... exit code -11 ...]

and nvargus-daemon logged

    NvPclOpen: PCL Open Failed. Error: 0xf
    SCF: Error BadParameter: Sensor could not be opened.

## 3. Why this is hardware, not the lean config

- The failure is in NVIDIA's capture layer (`nvargus-daemon`, sensor open),
  before the ZED SDK applies any depth/tracking/publish settings.
- After the daemon restart, the ZED X kernel driver re-probed the sensors
  inconsistently:

      zedx 14-0038: zedx_probe: Serial Number : 49910017
      zedx 14-0020: zedx_probe: Serial Number : 43779087
      zedx 14-0028: zedx_probe: Serial Number : 43779087   <- same serial twice
      (42569280, the front camera, not in the probe output)

- Before the power cycle, the same cameras streamed normally on the boot
  config (section 1).

GMSL links are brought up at Jetson boot; a camera-side power cycle while the
Jetson stays up needs a **Jetson reboot**. The boot stack was restarted
afterwards (`sudo systemctl start percept-stack`); its cameras are expected to
keep failing until that reboot.

## Still to do

After a Jetson reboot, with all three cameras streaming:

1. Re-take the baseline on the fresh boot stack (same script).
2. Stop the boot stack (no daemon restart needed after a fresh boot), run
   `perception_stack.launch.py`, and measure the same way.
3. Check each camera publishes only RGB, depth and confidence topics.
4. Run the launch a second time and confirm it reuses everything.
