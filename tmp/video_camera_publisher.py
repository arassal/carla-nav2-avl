#!/usr/bin/env python3
"""Publish three recorded camera videos as ROS 2 sensor_msgs/Image topics.

This is deliberately kept outside the ROS package so it can later be swapped
for a live-camera source without changing perception_costmap.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


CAMERAS = ("front", "left", "right")


class VideoCameraPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("recorded_camera_publisher")
        self.bridge = CvBridge()
        self.loop = args.loop
        self.captures = {
            "front": cv2.VideoCapture(args.front),
            "left": cv2.VideoCapture(args.left),
            "right": cv2.VideoCapture(args.right),
        }
        bad = [name for name, cap in self.captures.items() if not cap.isOpened()]
        if bad:
            raise RuntimeError("Could not open video(s): " + ", ".join(bad))

        self.source_fps = {
            name: cap.get(cv2.CAP_PROP_FPS) for name, cap in self.captures.items()
        }
        self.frame_counts = {
            name: int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for name, cap in self.captures.items()
        }
        invalid = [
            name for name in CAMERAS
            if self.source_fps[name] <= 0 or self.frame_counts[name] <= 0
        ]
        if invalid:
            raise RuntimeError("Invalid FPS/frame count for: " + ", ".join(invalid))
        self.offsets = {
            "front": args.front_offset,
            "left": args.left_offset,
            "right": args.right_offset,
        }
        if any(offset < 0 for offset in self.offsets.values()):
            raise ValueError("Video offsets must be non-negative")
        self.duration = min(
            self.frame_counts[name] / self.source_fps[name] - self.offsets[name]
            for name in CAMERAS
        )
        if self.duration <= 0:
            raise ValueError("A video offset is beyond the end of its recording")
        self.decoded_indices = {name: -1 for name in CAMERAS}
        self.latest_frames = {name: None for name in CAMERAS}
        self.playback_started = None

        self.image_publishers = {
            name: self.create_publisher(Image, f"/camera/{name}/image", qos_profile_sensor_data)
            for name in CAMERAS
        }
        self.info_publishers = {
            name: self.create_publisher(CameraInfo, f"/camera/{name}/camera_info",
                                        qos_profile_sensor_data)
            for name in CAMERAS
        }
        # The repository's geometry test uses 370 px as a ZED X-ish focal
        # length at 960 px width.  Per-device calibration is normally supplied
        # by the live ZED wrapper; recorded MP4s do not retain it.
        self.focal_px = args.focal_px
        if not 0.0 < args.scale <= 1.0:
            raise ValueError("--scale must be greater than 0 and at most 1")
        self.scale = args.scale
        # Publish at the fastest source rate. Slower recordings repeat their
        # latest frame; faster recordings are never accidentally slowed down.
        self.rate = args.rate if args.rate > 0 else min(self.source_fps.values())
        self.timer = self.create_timer(1.0 / self.rate, self.publish_frame_set)
        self.frame_index = 0
        self.get_logger().info(
            f"Publishing synchronized playback at {self.rate:.3f} Hz "
            f"for {self.duration:.3f} s; source FPS: "
            + ", ".join(f"{name}={self.source_fps[name]:.3f}" for name in CAMERAS)
            + "; topics: "
            + ", ".join(f"{name}=/camera/{name}/image" for name in CAMERAS)
        )

    def _rewind(self) -> None:
        for cap in self.captures.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.decoded_indices = {name: -1 for name in CAMERAS}
        self.latest_frames = {name: None for name in CAMERAS}
        self.frame_index = 0
        self.playback_started = time.monotonic()
        self.get_logger().info("Reached end of recording; looping to frame 0")

    def _frame_at(self, name: str, target_index: int):
        """Decode sequentially through target_index and return that frame.

        This avoids inaccurate random seeks in inter-frame-compressed MP4s.
        """
        cap = self.captures[name]
        while self.decoded_indices[name] < target_index:
            ok, frame = cap.read()
            if not ok:
                return None
            self.decoded_indices[name] += 1
            self.latest_frames[name] = frame
        return self.latest_frames[name]

    def publish_frame_set(self) -> None:
        now = time.monotonic()
        if self.playback_started is None:
            self.playback_started = now
        playback_time = now - self.playback_started
        if playback_time >= self.duration:
            if not self.loop:
                self.get_logger().info("Reached end of synchronized recording; stopping")
                rclpy.shutdown()
                return
            self._rewind()
            playback_time = 0.0

        targets = {
            name: min(
                int((playback_time + self.offsets[name]) * self.source_fps[name]),
                self.frame_counts[name] - 1,
            )
            for name in CAMERAS
        }
        frames = {name: self._frame_at(name, targets[name]) for name in CAMERAS}
        if any(frame is None for frame in frames.values()):
            self.get_logger().error("Could not decode a complete synchronized frame set")
            rclpy.shutdown()
            return

        stamp = self.get_clock().now().to_msg()
        for name, frame in frames.items():
            if self.scale != 1.0:
                frame = cv2.resize(
                    frame, None, fx=self.scale, fy=self.scale,
                    interpolation=cv2.INTER_AREA,
                )
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = f"{name}_camera"
            self.image_publishers[name].publish(msg)
            self.info_publishers[name].publish(self._camera_info(name, frame, stamp))
        self.frame_index += 1

    def _camera_info(self, name, frame, stamp) -> CameraInfo:
        height, width = frame.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = f"{name}_camera"
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        focal_px = self.focal_px * self.scale
        info.k = [focal_px, 0.0, cx,
                  0.0, focal_px, cy,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [focal_px, 0.0, cx, 0.0,
                  0.0, focal_px, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def destroy_node(self) -> bool:
        for cap in self.captures.values():
            cap.release()
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parents[1] / "videos" / "recording 2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", default=str(default_dir / "front.mp4"))
    parser.add_argument("--left", default=str(default_dir / "left.mp4"))
    parser.add_argument("--right", default=str(default_dir / "right.mp4"))
    parser.add_argument("--rate", type=float, default=0.0,
                        help="Frames/sec; 0 uses the front video's FPS")
    parser.add_argument("--focal-px", type=float, default=370.0,
                        help="Recorded-video focal length in pixels (default is approximate)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Image scale; intrinsics are scaled to match")
    parser.add_argument("--front-offset", type=float, default=0.0,
                        help="Seconds to skip at the start of the front video")
    parser.add_argument("--left-offset", type=float, default=0.0,
                        help="Seconds to skip at the start of the left video")
    parser.add_argument("--right-offset", type=float, default=0.0,
                        help="Seconds to skip at the start of the right video")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="Stop rather than replaying at end of video")
    parser.set_defaults(loop=True)
    return parser.parse_args()


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = VideoCameraPublisher(parse_args())
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
