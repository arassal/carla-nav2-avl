# Perception Scripts

Author: Adam Castillo ([@AdamCastillo07](https://github.com/AdamCastillo07))


- `costmap.py` — HSV color-thresholding road segmentation + obstacle
  detection + A* path planning. Satisfies "Image Thresholding &
  Color-based Lane/Feature Segmentation" task.
- `twinLiteNetTest.py` — learned (neural network) road/lane segmentation
  using TwinLiteNet+, for comparison against the classical approach.
  Experimental: depends on the external TwinLiteNetPlus repo + weights
  (see the file header) and has not been run end-to-end yet.

## Install

```bash
pip install -r requirements.txt
```

Optional dependencies (YOLO, torch/TwinLiteNet) are commented out in
`requirements.txt` — uncomment what you need.

## Run

```bash
# classical pipeline on the built-in synthetic demo image
python3 costmap.py --demo --out result.png

# on a real top-down image, dumping every intermediate stage
python3 costmap.py --image road.png --out result.png --debug
```

## Notes / known follow-ups

- HSV thresholds in `segment_road()` are tuned for the synthetic demo;
  recalibrate for real camera/lighting.
- A* currently plans left-edge → right-edge. Real BEV frames where the
  road runs top→bottom will need the planning direction made configurable.
- Standalone CLI scripts for now (image in → image out). Not yet wired
  into ROS2 as a node publishing `nav_msgs/OccupancyGrid`.
