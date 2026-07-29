#!/usr/bin/env python3
"""Plan a path N feet ahead, verify it clears the obstacle standing in the
lane, and only then execute it.

Rationale: a working detector does NOT imply a safe plan. In the CARLA phase
of this project perception correctly marked an obstacle while the global
planner still routed straight through it (nav2 StaticLayer was placing a
robot-relative grid only once, never re-placing it). The fix for that on this
vehicle -- costmap_odom_republisher.py -- has never been validated against
real hardware, and the thing standing in the lane right now is a person. So:
plan first, measure the plan's clearance from the obstacle, abort if it does
not clear, execute if it does.
"""
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import ComputePathToPose, NavigateToPose
import tf2_ros

FEET = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
DIST = FEET * 0.3048
MIN_CLEARANCE = 0.5      # robot_radius; plan must keep at least this from the person
LANE_HALF_W = 1.5        # what counts as "in the lane ahead"
OBST_THRESH = 97


class Runner(Node):
    def __init__(self):
        super().__init__("plan_then_go")
        self.buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf, self)
        self.grid = None
        self.odom = None
        self.create_subscription(OccupancyGrid, "/perception/costmap",
                                 lambda m: setattr(self, "grid", m),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odometry/filtered",
                                 lambda m: setattr(self, "odom", m), 10)
        self.plan_cli = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.nav_cli = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def spin(self, sec):
        t = time.time()
        while time.time() - t < sec:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    rclpy.init()
    n = Runner()
    n.spin(3.0)

    try:
        tf = n.buf.lookup_transform("odom", "base_link", rclpy.time.Time())
    except Exception as e:
        print(f"ABORT: no TF ({e})"); return 1
    t = tf.transform.translation; q = tf.transform.rotation
    yaw = 2.0 * math.atan2(q.z, q.w)
    sx, sy = t.x, t.y
    gx = sx + DIST * math.cos(yaw)
    gy = sy + DIST * math.sin(yaw)
    print(f"[go] start odom=({sx:.2f},{sy:.2f}) yaw={math.degrees(yaw):.1f}deg")
    print(f"[go] goal  {FEET:.0f} ft -> odom=({gx:.2f},{gy:.2f})")

    # --- obstacle cells in the lane ahead, converted base_link -> odom ---
    if n.grid is None:
        print("ABORT: no costmap"); return 1
    # Raycast to the FIRST occupied cell per bearing, exactly as
    # costmap_to_cloud does. Taking every occupied cell instead would sweep in
    # the occlusion shadow behind an obstacle (perception marks it lethal too),
    # and since any forward path necessarily crosses that shadow the clearance
    # test would report ~0 m and veto every plan -- which it did.
    import numpy as np
    g = n.grid; r = g.info.resolution
    ox, oy = g.info.origin.position.x, g.info.origin.position.y
    W, H = g.info.width, g.info.height
    grid = np.asarray(g.data, dtype=np.int8).reshape(H, W)
    A = np.radians(np.arange(-100.0, 100.5, 0.5))
    R = np.arange(0.3, min(DIST + 2, 15.0), 0.05)
    GX = np.cos(A)[:, None] * R[None, :]
    GY = np.sin(A)[:, None] * R[None, :]
    gix = ((GX - ox) / r).astype(np.int32)
    giy = ((GY - oy) / r).astype(np.int32)
    okm = (gix >= 0) & (gix < W) & (giy >= 0) & (giy < H)
    vv = np.full(GX.shape, -1, dtype=np.int16)
    vv[okm] = grid[giy[okm], gix[okm]]
    hitm = vv >= OBST_THRESH
    rows_h = np.nonzero(hitm.any(axis=1))[0]
    cols_h = hitm.argmax(axis=1)[rows_h]
    obst = []
    for a_i, r_i in zip(rows_h, cols_h):
        bx = float(GX[a_i, r_i]); by = float(GY[a_i, r_i])
        if 0.3 < bx and abs(by) < LANE_HALF_W:
            obst.append((sx + bx * math.cos(yaw) - by * math.sin(yaw),
                         sy + bx * math.sin(yaw) + by * math.cos(yaw)))
    print(f"[go] obstacle cells in lane ahead: {len(obst)}")
    if obst:
        nd = min(math.hypot(o[0] - sx, o[1] - sy) for o in obst)
        print(f"[go] nearest obstacle: {nd:.2f} m ahead")

    # --- plan only, do not move ---
    if not n.plan_cli.wait_for_server(timeout_sec=10.0):
        print("ABORT: planner server unavailable"); return 1
    pg = ComputePathToPose.Goal()
    pg.goal = PoseStamped()
    pg.goal.header.frame_id = "odom"
    pg.goal.pose.position.x = gx
    pg.goal.pose.position.y = gy
    pg.goal.pose.orientation.z = q.z
    pg.goal.pose.orientation.w = q.w
    pg.use_start = False
    fut = n.plan_cli.send_goal_async(pg)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
    h = fut.result()
    if h is None or not h.accepted:
        print("ABORT: plan request rejected"); return 1
    rf = h.get_result_async()
    rclpy.spin_until_future_complete(n, rf, timeout_sec=15.0)
    res = rf.result()
    path = res.result.path.poses if res else []
    if not path:
        print("ABORT: planner returned EMPTY path (no route around obstacle)"); return 1

    # --- does the plan clear the person? ---
    lat = []
    for p in path:
        dx = p.pose.position.x - sx; dy = p.pose.position.y - sy
        lat.append(-dx * math.sin(yaw) + dy * math.cos(yaw))
    clr = None
    if obst:
        clr = min(math.hypot(p.pose.position.x - o[0], p.pose.position.y - o[1])
                  for p in path for o in obst)
    print(f"[go] path: {len(path)} poses, lateral swing {min(lat):+.2f}..{max(lat):+.2f} m")
    print(f"[go] closest the PLAN comes to the obstacle: "
          f"{f'{clr:.2f} m' if clr is not None else 'n/a (no obstacle)'}")

    if clr is not None and clr < MIN_CLEARANCE:
        print(f"*** ABORT: plan passes {clr:.2f} m from the person "
              f"(< {MIN_CLEARANCE} m). NOT DRIVING. ***")
        return 2
    print(f"[go] plan clears the obstacle -- EXECUTING")

    # --- execute ---
    if not n.nav_cli.wait_for_server(timeout_sec=10.0):
        print("ABORT: nav server unavailable"); return 1
    ng = NavigateToPose.Goal()
    ng.pose = pg.goal
    ng.pose.header.stamp = n.get_clock().now().to_msg()
    nf = n.nav_cli.send_goal_async(ng)
    rclpy.spin_until_future_complete(n, nf, timeout_sec=10.0)
    nh = nf.result()
    if nh is None or not nh.accepted:
        print("ABORT: nav goal rejected"); return 1
    print("[go] goal ACCEPTED -- driving\n")

    rf2 = nh.get_result_async()
    t0 = time.time(); last = 0.0
    while time.time() - t0 < 90:
        rclpy.spin_once(n, timeout_sec=0.1)
        if rf2.done():
            break
        if n.odom and time.time() - last > 1.0:
            last = time.time()
            p = n.odom.pose.pose.position
            v = n.odom.twist.twist.linear
            d = math.hypot(p.x - sx, p.y - sy)
            latnow = -(p.x - sx) * math.sin(yaw) + (p.y - sy) * math.cos(yaw)
            print(f"   t={time.time()-t0:5.1f}s  travelled={d:5.2f}m  "
                  f"lateral={latnow:+.2f}m  v={math.hypot(v.x,v.y):.2f} m/s")
    p = n.odom.pose.pose.position if n.odom else None
    if p:
        print(f"\n[go] FINAL travelled {math.hypot(p.x-sx, p.y-sy):.2f} m "
              f"({math.hypot(p.x-sx, p.y-sy)/0.3048:.1f} ft)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
