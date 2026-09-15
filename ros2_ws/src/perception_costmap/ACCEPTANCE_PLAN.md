# Dinosaur Perception Acceptance Plan

Scope: three ZED X cameras, road segmentation, YOLO/cone detection, metric
depth projection, temporal fusion, `/perception/costmap`, and the Nav2
`/perception/costmap_cloud` bridge. Blind-region policy is unchanged.

## Completed Remotely

- Bounded asynchronous inference; front camera every cycle and alternating
  side-camera coverage without detector backlog.
- Timestamp-buffered matching of RGB, registered ZED depth, and SDK confidence.
- Wheel-odometry interpolation at each image capture timestamp, followed by
  reprojection to the current robot pose.
- Confidence filtering, detection-edge erosion, per-component median/MAD depth
  rejection, finite/range gates, and explicit IPM fallback.
- Required-front-camera and required-detector freshness gates.
- ROS diagnostics for every camera, detector inference, and costmap freshness.
- Pure, tested nearest-hit conversion from costmap cells to Nav2 ray endpoints.
- Class-specific hard exclusion zones that remain lethal through the Nav2
  cloud bridge, plus softer outer gradients for operator interpretation.
- Class-specific temporal policy: people and vehicles mark on one confident
  result; people clear more conservatively than vehicles and generic blobs.
- Operator RViz rendering separates unobserved, free, caution, off-road,
  lethal, and the exact collision boundary consumed by Nav2.
- ZED timestamp and depth-quality measurement tools plus complete bag recorder.

## Measured Baseline

| Gate | Result | Acceptance |
|---|---:|---:|
| Unit/offline tests | 78 passing | all pass |
| Costmap publication | 10.0 Hz | >=9.5 Hz |
| Steady inference backlog | no growth after warm-up | no growth |
| RGB/depth timestamp p50 | 0 ms, all cameras | <=10 ms |
| Valid ZED depth | 92.7% front, 84.4% left, 91.3% right | >=80% stationary |
| Depth retained at confidence 70 | 94.8%, 91.9%, 93.3% | >=90% of valid depth |
| 60-second GPU temperature | 50-51 C | <80 C |
| 60-second CPU temperature | 55-57 C | <85 C |
| Swap use | 0 MB | 0 MB |
| RViz CPU after visualization throttle | ~42% (was ~166%) | debug UI must not starve control |
| Costmap colorizer CPU | ~26% (was ~88%) | debug UI must not starve control |

## Remaining Evidence Gates

1. Record representative campus SVOs/bags: sun, shade, glare, wet pavement,
   slopes, grass/curbs, cones, people, bicycles, and parked/moving vehicles.
   Use ZED SVO for long camera runs; full synchronized ROS bags are roughly
   60 MiB/s before compression and should be limited to short test scenarios.
2. Label road masks and obstacle ground positions. Require road IoU >=0.90,
   road precision >=0.98, and zero false-drivable samples in the safety set.
3. Survey calibration targets from 1-12 m. Require <=0.15 m projection RMSE
   through 8 m and <=0.35 m from 8-12 m for each camera.
4. Measure person/vehicle/cone precision, recall, range error, and clearing time;
   tune thresholds only from precision-recall curves, never visual preference.
5. Replay every accepted bag and compare costmap hashes/metrics against the
   frozen baseline. Reject regressions in detection recall, projection error,
   stale output, latency, or false-drivable area.
6. Run a 30-minute stationary full-stack thermal soak and injected camera,
   depth, confidence, detector, and odometry failures with motors disabled.
7. Run supervised low-speed closed-course tests, measure stopping distance and
   planner clearance, then derive speed-aware safety margins.
8. Freeze model hashes, camera serials, ZED settings, calibration files, ROS
   parameters, and the final acceptance report for competition use.

No software-only result can satisfy gates 1-4 or 7 without labeled/surveyed
field evidence. Autonomous operation is not accepted until those gates pass.
