#!/usr/bin/env python3
"""Measure ZED RGB/depth/confidence timestamp alignment without moving the car."""

import argparse
from collections import defaultdict
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def stamp_seconds(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def alignment_metrics(reference, candidates, tolerance):
    if not reference or not candidates:
        return {"samples": 0, "matched_fraction": 0.0}
    candidate_array = np.asarray(candidates)
    deltas = np.asarray([
        np.min(np.abs(candidate_array - stamp)) for stamp in reference
    ])
    return {
        "samples": int(len(deltas)),
        "matched_fraction": float(np.mean(deltas <= tolerance)),
        "delta_ms_p50": float(1000.0 * np.percentile(deltas, 50)),
        "delta_ms_p95": float(1000.0 * np.percentile(deltas, 95)),
        "delta_ms_max": float(1000.0 * np.max(deltas)),
    }


class StampCollector(Node):
    def __init__(self, cameras):
        super().__init__("measure_zed_sync")
        self.stamps = defaultdict(list)
        self._image_subscriptions = []
        for camera in cameras:
            base = f"/zed_{camera}/zed_node"
            topics = {
                "rgb": f"{base}/rgb/color/rect/image",
                "depth": f"{base}/depth/depth_registered",
                "confidence": f"{base}/confidence/confidence_map",
            }
            for stream, topic in topics.items():
                key = (camera, stream)
                self._image_subscriptions.append(self.create_subscription(
                    Image, topic,
                    lambda message, sample_key=key: self.stamps[sample_key].append(
                        stamp_seconds(message)),
                    qos_profile_sensor_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="+", default=["front", "left", "right"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    rclpy.init()
    node = StampCollector(args.cameras)
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    tolerance = args.tolerance_ms / 1000.0
    report = {"duration_sec": args.duration, "tolerance_ms": args.tolerance_ms,
              "cameras": {}}
    for camera in args.cameras:
        rgb = node.stamps[(camera, "rgb")]
        report["cameras"][camera] = {
            "counts": {
                stream: len(node.stamps[(camera, stream)])
                for stream in ("rgb", "depth", "confidence")
            },
            "rgb_depth": alignment_metrics(
                rgb, node.stamps[(camera, "depth")], tolerance),
            "rgb_confidence": alignment_metrics(
                rgb, node.stamps[(camera, "confidence")], tolerance),
        }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")


if __name__ == "__main__":
    main()
