#!/usr/bin/env python3
"""Show the three ROS camera image topics in one OpenCV window.

Press q or Escape to close.  The program is only a subscriber: replacing the
recorded-video publisher with live cameras does not require changing this file.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


CAMERAS = ("front", "left", "right")


class CameraDisplay(Node):
    def __init__(self, width: int, sync_slop_ms: float) -> None:
        super().__init__("three_camera_display")
        self.bridge = CvBridge()
        self.width = width
        self.sync_slop_ns = int(sync_slop_ms * 1_000_000)
        self.latest: dict[str, tuple[int, np.ndarray]] = {}
        self.synced_frames: dict[str, np.ndarray] = {}
        self.last_stamp_set: tuple[int, ...] | None = None
        self.sync_skew_ms = 0.0
        for name in CAMERAS:
            self.create_subscription(
                Image, f"/camera/{name}/image",
                lambda msg, camera=name: self._on_image(camera, msg),
                qos_profile_sensor_data,
            )
        cv2.namedWindow("Raw camera feeds", cv2.WINDOW_NORMAL)
        self.create_timer(1.0 / 30.0, self.render)
        self.get_logger().info("Listening to /camera/{front,left,right}/image")

    def _on_image(self, camera: str, msg: Image) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest[camera] = (stamp_ns, frame)
        if not all(name in self.latest for name in CAMERAS):
            return
        stamps = tuple(self.latest[name][0] for name in CAMERAS)
        spread = max(stamps) - min(stamps)
        if spread > self.sync_slop_ns or stamps == self.last_stamp_set:
            return
        self.synced_frames = {name: self.latest[name][1] for name in CAMERAS}
        self.last_stamp_set = stamps
        self.sync_skew_ms = spread / 1_000_000.0

    @staticmethod
    def _tile(image: np.ndarray | None, label: str, width: int) -> np.ndarray:
        if image is None:
            tile = np.zeros((int(width * 9 / 16), width, 3), dtype=np.uint8)
            text = f"Waiting for {label}"
        else:
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            tile = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            text = label
        cv2.rectangle(tile, (0, 0), (max(180, len(text) * 12), 32), (0, 0, 0), -1)
        cv2.putText(tile, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return tile

    def render(self) -> None:
        tile_width = self.width // 2
        suffix = f" (sync spread {self.sync_skew_ms:.1f} ms)"
        front = self._tile(self.synced_frames.get("front"), "front" + suffix, tile_width)
        left = self._tile(self.synced_frames.get("left"), "left" + suffix, tile_width)
        right = self._tile(self.synced_frames.get("right"), "right" + suffix, tile_width)
        blank = np.zeros_like(right)
        mosaic = np.vstack((np.hstack((front, left)), np.hstack((right, blank))))
        cv2.imshow("Raw camera feeds", mosaic)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1440,
                        help="Total mosaic width in pixels")
    parser.add_argument("--sync-slop-ms", type=float, default=40.0,
                        help="Maximum timestamp spread for a displayed frame set")
    args = parser.parse_args()
    rclpy.init()
    node = CameraDisplay(args.width, args.sync_slop_ms)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
