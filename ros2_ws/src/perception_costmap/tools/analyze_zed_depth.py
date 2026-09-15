#!/usr/bin/env python3
"""Measure confidence-filtered ZED depth coverage from synchronized frames."""

import argparse
from collections import defaultdict
import json
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def stamp_key(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class DepthQuality(Node):
    def __init__(self, cameras, thresholds):
        super().__init__("analyze_zed_depth")
        self.bridge = CvBridge()
        self.thresholds = thresholds
        self.pending = defaultdict(lambda: {"depth": {}, "confidence": {}})
        self.records = defaultdict(list)
        self._image_subscriptions = []
        for camera in cameras:
            base = f"/zed_{camera}/zed_node"
            self._image_subscriptions.extend([
                self.create_subscription(
                    Image, f"{base}/depth/depth_registered",
                    lambda msg, c=camera: self._receive(c, "depth", msg),
                    qos_profile_sensor_data),
                self.create_subscription(
                    Image, f"{base}/confidence/confidence_map",
                    lambda msg, c=camera: self._receive(c, "confidence", msg),
                    qos_profile_sensor_data),
            ])

    def _receive(self, camera, stream, message):
        key = stamp_key(message)
        encoding = "32FC1" if stream == "depth" else "passthrough"
        array = self.bridge.imgmsg_to_cv2(
            message, desired_encoding=encoding).astype(np.float32, copy=False)
        self.pending[camera][stream][key] = array
        other = "confidence" if stream == "depth" else "depth"
        if key not in self.pending[camera][other]:
            self._trim(camera)
            return
        depth = self.pending[camera]["depth"].pop(key)
        confidence = self.pending[camera]["confidence"].pop(key)
        valid = np.isfinite(depth) & (depth >= 0.3) & (depth <= 15.0)
        record = {"valid_depth_fraction": float(np.mean(valid))}
        if valid.any():
            values = confidence[valid]
            record["confidence_p50"] = float(np.percentile(values, 50))
            record["confidence_p95"] = float(np.percentile(values, 95))
            for threshold in self.thresholds:
                record[f"retained_at_{threshold}"] = float(
                    np.mean(values <= threshold))
        self.records[camera].append(record)
        self._trim(camera)

    def _trim(self, camera):
        for stream in ("depth", "confidence"):
            samples = self.pending[camera][stream]
            for key in sorted(samples)[:-8]:
                samples.pop(key, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="+", default=["front", "left", "right"])
    parser.add_argument("--thresholds", nargs="+", type=int,
                        default=[30, 50, 70, 90])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    rclpy.init()
    node = DepthQuality(args.cameras, args.thresholds)
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    report = {"duration_sec": args.duration, "cameras": {}}
    for camera in args.cameras:
        records = node.records[camera]
        summary = {"frames": len(records)}
        if records:
            for key in records[0]:
                summary[key + "_mean"] = float(np.mean([
                    record[key] for record in records if key in record]))
        report["cameras"][camera] = summary
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")


if __name__ == "__main__":
    main()
