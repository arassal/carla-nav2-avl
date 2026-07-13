# Recorded-video costmap viewer

## Recommended: start everything once

Inside the `ubuntu` distrobox:

```zsh
cd ~/Projects/carla-nav2-avl
zsh tmp/run_video_test.zsh
```

I ran this from feature/alexander branch. What i called this folder/directory was tmp but i don't think that matters. A sister video folder should exist though you could easily get away with making a video folder within this directory and adjusting the code to reflect the change.

Keep that terminal open. Press Ctrl-C in it to stop the complete test. The
launcher refuses to start if a `perception_costmap` node is already running,
which prevents multiple configurations from publishing interleaved maps.

The recordings have different native rates (front about 8 FPS, sides about
15 FPS). The publisher aligns them by elapsed video time and outputs matched
sets at the common 8 FPS review rate. If the recordings themselves began at
different real-world moments, adjust `--front-offset`, `--left-offset`, or
`--right-offset` on `video_camera_publisher.py`; each value skips that many
seconds from the beginning of that recording.

The launcher uses half-resolution images for real-time costmap processing and
scales `CameraInfo` intrinsics with them, so the projection geometry is
unchanged. The marker view applies a three-frame visual majority filter to
reduce single-frame segmentation shimmer; it does not alter Nav2's original
`/perception/costmap` data.

Arrange the OpenCV window and RViz side by side. The publisher can later be
replaced by live camera nodes as long as they publish the same three topics.

The camera mounts in `video_costmap.yaml` match the project’s real ZED X
installation. The MP4s have no per-device ZED calibration attached, so the
publisher currently uses an approximate 370 px focal length. Live ZED wrapper
topics supply the actual `CameraInfo` automatically; use that calibration
before treating the position or shape of the costmap as physically accurate.

`video_costmap_markers.rviz` intentionally uses a Marker display instead of
RViz's Map display. It avoids the `indexed_8bit_image` GLSL error seen on this
machine; the original `video_costmap.rviz` remains available for systems where
the Map display renders normally.
