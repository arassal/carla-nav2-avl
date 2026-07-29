#!/usr/bin/env python3
"""Independent hard distance cap: E-stop the car if it travels further than
--limit meters from wherever it was when this node started.

This is a SEPARATE safety layer from (a) the webui E-stop, which needs a human
to react, and (b) Nav2's own goal tolerance, which is only as trustworthy as
Nav2. If Nav2 misbehaves -- bad goal, runaway controller, odom glitch sending
it past the intended stopping point -- this trips automatically.

It publishes estop ONLY on trip, never continuously: actuator_node ranks
/avros/actuator_command ABOVE /cmd_vel, so a node publishing actuator_command
on a timer would silently override Nav2 for as long as it ran. One-shot on
trip only, then latched by actuator_node's own _estop flag.
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from avros_msgs.msg import ActuatorCommand


class DistanceWatchdog(Node):
    def __init__(self, limit):
        super().__init__("distance_watchdog")
        self.limit = limit
        self.start = None
        self.tripped = False
        self.pub = self.create_publisher(
            ActuatorCommand, "/avros/actuator_command", 10)
        self.create_subscription(
            Odometry, "/odometry/filtered", self.cb, 10)
        self.get_logger().warn(
            f"distance watchdog ARMED: will E-STOP beyond {limit:.2f} m "
            f"({limit / 0.3048:.1f} ft) from start")

    def cb(self, msg):
        p = msg.pose.pose.position
        if self.start is None:
            self.start = (p.x, p.y)
            self.get_logger().info(
                f"start reference = odom ({p.x:.2f}, {p.y:.2f})")
            return
        if self.tripped:
            return

        d = math.hypot(p.x - self.start[0], p.y - self.start[1])
        if d > self.limit:
            self.tripped = True
            cmd = ActuatorCommand()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.estop = True
            cmd.throttle = 0.0
            cmd.brake = 1.0
            cmd.steer = 0.0
            cmd.mode = "N"
            self.pub.publish(cmd)
            self.get_logger().error(
                f"DISTANCE LIMIT EXCEEDED ({d:.2f} m > {self.limit:.2f} m) "
                f"-- E-STOP PUBLISHED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=7.0,
                    help="metres from start before E-stop (default 7.0)")
    args = ap.parse_args()

    rclpy.init()
    node = DistanceWatchdog(args.limit)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Ctrl-C / SIGTERM is a normal way to end the watchdog -- exit quietly
        # so a clean stop is never mistaken for a crash mid-test.
        node.get_logger().info("watchdog stopped")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
