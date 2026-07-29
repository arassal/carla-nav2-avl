#!/usr/bin/env python3
"""Drive straight forward a fixed distance, closed-loop on EKF odometry.

Deliberately does NOT use Nav2 -- raw /cmd_vel only, no planner, no costmap,
NO OBSTACLE AVOIDANCE. The path must be verified clear by a human first.

Stopping accounts for the actuator's own slew limits (accel 0.3 m/s^2,
decel 1.3 m/s^2 from actuator_params.yaml): the command is cut early by the
predicted coast distance v^2/(2*decel) so the vehicle settles at the target
rather than overshooting past it.

Usage: drive_forward.py [feet] [--mps SPEED]
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from avros_msgs.msg import ActuatorCommand

DECEL = 1.3          # max_linear_decel_mps2, actuator_params.yaml
HARD_MARGIN = 0.5    # abort if we somehow exceed target by this much


class Driver(Node):
    def __init__(self, target_m, speed):
        super().__init__("drive_forward")
        self.target = target_m
        self.speed = speed
        self.start = None
        self.pos = None
        self.v = 0.0
        self.done = False
        self.reason = ""

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.act_pub = self.create_publisher(
            ActuatorCommand, "/avros/actuator_command", 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.odom_cb, 10)

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pos = (p.x, p.y)
        if self.start is None:
            self.start = (p.x, p.y)
        lv = msg.twist.twist.linear
        self.v = math.hypot(lv.x, lv.y)

    def dist(self):
        if self.start is None or self.pos is None:
            return 0.0
        return math.hypot(self.pos[0] - self.start[0], self.pos[1] - self.start[1])

    def send(self, v):
        t = Twist()
        t.linear.x = v
        t.angular.z = 0.0
        self.cmd_pub.publish(t)

    def estop(self):
        c = ActuatorCommand()
        c.header.stamp = self.get_clock().now().to_msg()
        c.estop = True
        c.throttle = 0.0
        c.brake = 1.0
        c.steer = 0.0
        c.mode = "N"
        self.act_pub.publish(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feet", nargs="?", type=float, default=10.0)
    ap.add_argument("--mps", type=float, default=0.447,
                    help="forward speed m/s (default 0.447 = 1 mph)")
    args = ap.parse_args()

    target = args.feet * 0.3048
    rclpy.init()
    node = Driver(target, args.mps)

    # wait for odom so we have a valid start reference
    t0 = time.time()
    while node.start is None and time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.start is None:
        print("[drive] ABORT: no odometry", flush=True)
        return

    print(f"[drive] target {args.feet:.1f} ft = {target:.3f} m "
          f"at {args.mps:.3f} m/s ({args.mps / 0.447:.1f} mph)", flush=True)
    print(f"[drive] start odom = ({node.start[0]:.3f}, {node.start[1]:.3f})",
          flush=True)

    rate_hz = 20.0
    t_start = time.time()
    last_log = 0.0
    samples = []   # (t, distance, actual speed) -- velocity profile for tuning
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0 / rate_hz)
            d = node.dist()
            coast = (node.v ** 2) / (2.0 * DECEL)

            if d >= target + HARD_MARGIN:
                node.reason = f"OVERSHOOT GUARD ({d:.3f} m)"
                break
            if d + coast >= target:
                node.reason = f"target reached (cut at {d:.3f} m, coast {coast:.3f} m)"
                break
            # 10 ft at 1 mph is ~8 s including ramps; if it hasn't arrived well
            # inside 25 s something is wrong (blocked wheels, or manual control
            # outranking /cmd_vel) -- stop rather than keep pushing.
            if time.time() - t_start > 25:
                node.reason = (f"TIMEOUT (25 s, only {d:.3f} m) -- is the webui "
                               f"streaming manual commands? press AUTONOMOUS")
                break

            node.send(args.mps)
            samples.append((time.time() - t_start, d, node.v))
            if time.time() - last_log > 0.5:
                last_log = time.time()
                print(f"[drive]  t={time.time() - t_start:5.2f}s  {d:5.3f}/{target:.3f} m"
                      f"   v={node.v:.3f} (cmd {args.mps:.3f}) m/s", flush=True)
    except KeyboardInterrupt:
        node.reason = "INTERRUPTED"

    # stop commanding, then hold zeros through the actuator's decel ramp
    print(f"[drive] stopping -- {node.reason}", flush=True)
    for _ in range(15):
        node.send(0.0)
        rclpy.spin_once(node, timeout_sec=0.05)

    t1 = time.time()
    while time.time() - t1 < 2.5:
        rclpy.spin_once(node, timeout_sec=0.05)

    print(f"[drive] FINAL distance = {node.dist():.3f} m "
          f"({node.dist() / 0.3048:.2f} ft), speed {node.v:.3f} m/s", flush=True)

    # --- velocity profile summary (the data that decides whether the slew /
    # PID gains actually need touching, vs the motion just being commanded
    # badly) ---
    if samples:
        vs = [s[2] for s in samples]
        vmax = max(vs)
        cruise = [v for v in vs if v > 0.8 * args.mps]
        t90 = next((t for t, _, v in samples if v >= 0.9 * args.mps), None)
        print(f"[prof] commanded      {args.mps:.3f} m/s", flush=True)
        print(f"[prof] peak actual    {vmax:.3f} m/s  ({vmax / args.mps * 100:.0f}% of cmd)",
              flush=True)
        if cruise:
            mean_c = sum(cruise) / len(cruise)
            jitter = max(cruise) - min(cruise)
            print(f"[prof] cruise mean    {mean_c:.3f} m/s over {len(cruise)} samples",
                  flush=True)
            print(f"[prof] cruise jitter  {jitter:.3f} m/s "
                  f"(high = rough/fighting, low = smooth)", flush=True)
        else:
            print("[prof] never reached 80% of commanded speed", flush=True)
        print(f"[prof] time to 90%    "
              f"{f'{t90:.2f} s' if t90 is not None else 'never reached'}", flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
