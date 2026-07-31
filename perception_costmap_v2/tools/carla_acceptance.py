#!/usr/bin/env python3
"""
carla_acceptance.py -- DEPLOY.md section 6's acceptance checklist, measured
instead of eyeballed in RViz. Runs on the CARLA box with tools/carla_feed.py
and the costmap node already running.

The checklist says "a pedestrian standing in view goes lethal within ~300ms
and clears within ~500ms after stepping away". Watching that in RViz tells
you whether it happened, not how long it took, and 300 vs 800 ms looks
identical to a human. This spawns a walker at a known offset in front of the
ego, subscribes to /perception/costmap, and timestamps the transitions.

The ego is put in handbrake for the run: with autopilot driving, the walker
leaves the field of view mid-measurement and the clear-time reading becomes
"how fast did we drive past it" rather than the temporal filter's decay.

Usage:
    # terminal 1: CarlaUE4.sh     terminal 2: tools/carla_feed.py
    # terminal 3: costmap_node    terminal 4:
    PYTHONPATH=. python3 tools/carla_acceptance.py
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def find_ego(world):
    """The vehicle carla_feed spawned. There is exactly one in this setup;
    if a scenario ever runs more, pick the one with a camera attached."""
    vehicles = [a for a in world.get_actors() if a.type_id.startswith("vehicle.")]
    if not vehicles:
        raise SystemExit("no vehicle in the world -- is tools/carla_feed.py running?")
    return vehicles[0]


class CostmapWatcher:
    """Tracks whether a metric region of /perception/costmap reads lethal,
    with the receive timestamp of each message."""

    def __init__(self, node, x, y, radius_m, lethal=100):
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        self.x, self.y, self.radius_m, self.lethal = x, y, radius_m, lethal
        self.samples = []          # (recv_time, lethal_count_in_region)
        node.create_subscription(OccupancyGrid, "/perception/costmap", self._cb, qos)

    def _cb(self, msg):
        info = msg.info
        res = info.resolution
        data = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        col = int((self.x - info.origin.position.x) / res)
        row = int((self.y - info.origin.position.y) / res)
        r = max(1, int(self.radius_m / res))
        sub = data[max(0, row - r):row + r + 1, max(0, col - r):col + r + 1]
        self.samples.append((time.time(), int((sub >= self.lethal).sum())))

    def wait_for(self, node, predicate, timeout, t0):
        """Return seconds from t0 until the newest sample satisfies
        predicate(lethal_count), or None on timeout.

        Spinning here is not optional: without it no callback ever fires, the
        newest sample stays frozen at whatever it was on entry, and this
        reports a timeout no matter what the costmap actually did.
        """
        import rclpy
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            if self.samples and predicate(self.samples[-1][1]):
                return self.samples[-1][0] - t0
        return None

    def series_since(self, t0):
        return [(s - t0, c) for s, c in self.samples if s >= t0]


def _fmt_series(series, limit=8):
    """The raw per-message counts behind a latency number -- a single 'PASS'
    hides whether the transition was clean or bounced around the threshold."""
    return " ".join("%dms:%d" % (round(dt * 1000), n) for dt, n in series[:limit])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--distance", type=float, default=6.0,
                    help="metres ahead of the ego to place the walker")
    ap.add_argument("--radius", type=float, default=1.0,
                    help="metres around that point counted as 'the obstacle'")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    import carla
    import rclpy

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    ego = find_ego(world)

    rclpy.init()
    node = rclpy.create_node("carla_acceptance")

    # Stop the ego so the geometry stays fixed for the whole measurement.
    ego.set_autopilot(False)
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

    watcher = CostmapWatcher(node, args.distance, 0.0, args.radius)

    def spin(seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    print("settling (ego braked, no walker)...")
    spin(2.0)
    if not watcher.samples:
        raise SystemExit("no /perception/costmap messages -- is the node running?")
    baseline = watcher.samples[-1][1]
    rate = len(watcher.samples) / 2.0
    print("  /perception/costmap rate      : %.2f Hz   %s"
          % (rate, "PASS (>= 8)" if rate >= 8.0 else "FAIL (< 8)"))
    print("  lethal cells in region before : %d" % baseline)

    tf = ego.get_transform()
    fwd = tf.get_forward_vector()
    loc = carla.Location(x=tf.location.x + fwd.x * args.distance,
                         y=tf.location.y + fwd.y * args.distance,
                         z=tf.location.z + 1.0)
    walker_bp = world.get_blueprint_library().filter("walker.pedestrian.*")[0]

    walker = None
    try:
        print("\nspawning walker %.1f m ahead..." % args.distance)
        t0 = time.time()
        walker = world.try_spawn_actor(walker_bp, carla.Transform(loc, tf.rotation))
        if walker is None:
            raise SystemExit("walker spawn failed (blocked?) -- try --distance")

        del watcher.samples[:]
        appear = watcher.wait_for(node, lambda n: n > baseline, args.timeout, t0)
        print("  time to lethal                : %s   %s"
              % ("%.0f ms" % (appear * 1000) if appear else "NOT DETECTED",
                 ("PASS (<= 300)" if appear and appear <= 0.3 else "FAIL")))
        print("    " + _fmt_series(watcher.series_since(t0)))

        spin(1.5)
        held = watcher.samples[-1][1]
        print("  lethal cells while present    : %d" % held)

        print("\ndestroying walker...")
        t1 = time.time()
        walker.destroy()
        walker = None
        del watcher.samples[:]
        clear = watcher.wait_for(node, lambda n: n <= baseline, args.timeout, t1)
        print("  time to clear                 : %s   %s"
              % ("%.0f ms" % (clear * 1000) if clear else "NEVER CLEARED",
                 ("PASS (<= 500)" if clear and clear <= 0.5 else "FAIL")))
        print("    " + _fmt_series(watcher.series_since(t1)))
    finally:
        if walker is not None:
            try: walker.destroy()
            except Exception: pass
        ego.apply_control(carla.VehicleControl(hand_brake=False))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
