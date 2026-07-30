#!/usr/bin/env python3
"""
carla_feed.py -- REAL CARLA integration. Run this ONLY on a machine with
CARLA installed and a discrete GPU (CARLA 0.9.16, UE4-based; this repo's
core pipeline is otherwise CARLA-version-agnostic). This sandbox has no GPU
and cannot run this script -- see tools/synthetic_demo.py for a data-free
stand-in that exercises the same downstream pipeline.

What this does:
  1. Connects to a running CARLA server (`./CarlaUE4.sh`), spawns an
     autopilot-driven ego vehicle.
  2. Attaches an RGB camera, a semantic-segmentation camera (same transform,
     for ground truth), and a lidar.
  3. Publishes ROS2 topics the perception node consumes directly -- this
     bypasses carla-ros-bridge, which historically lags the CARLA Python
     API version and is a common source of "why is nothing publishing"
     sim-to-real bugs.
  4. Optionally dumps paired RGB/semantic PNGs (--dump-dir) for
     tools/eval_road_iou.py to score segmenters against.

Usage:
    python3 tools/carla_feed.py --host 127.0.0.1 --port 2000 \
        --dump-dir /tmp/carla_eval_frames --dump-every 10

Requires: carla==0.9.16 (`pip install carla==0.9.16`, matched to your
server binary), rclpy + sensor_msgs (a sourced ROS2 Humble environment).
Both imports are lazy so importing this module elsewhere never requires
either.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from perception_costmap.carla_convert import (
    bgra_bytes_to_bgr, carla_lidar_to_rep103, semantic_to_road_mask,
)

IMG_W, IMG_H, FOV = 640, 480, 90
LIDAR_MOUNT_Z = 1.8
CAMERA_MOUNT_Z = 1.6


def _require_carla():
    try:
        import carla
        return carla
    except ImportError as e:
        raise SystemExit(
            "carla package not importable. This script only runs on the "
            "x86 sim box with CARLA installed: `pip install carla==0.9.16` "
            "(match your CarlaUE4 server version), and CarlaUE4.sh must "
            "already be running on --host:--port."
        ) from e


def _require_ros2():
    try:
        import rclpy
        from sensor_msgs.msg import Image, PointCloud2, PointField
        return rclpy, Image, PointCloud2, PointField
    except ImportError as e:
        raise SystemExit(
            "rclpy/sensor_msgs not importable. Source a ROS2 Humble "
            "environment first: `source /opt/ros/humble/setup.bash`."
        ) from e


def build_pointcloud2(points_xyz, frame_id, stamp, PointCloud2, PointField):
    import struct
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = points_xyz.shape[0]
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None, help="e.g. Town01 -- reloads the world if set")
    ap.add_argument("--dump-dir", default=None, help="if set, dump paired RGB/semantic PNGs here")
    ap.add_argument("--dump-every", type=int, default=10, help="dump every Nth tick")
    ap.add_argument("--ticks", type=int, default=0, help="0 = run until Ctrl-C")
    args = ap.parse_args()

    carla = _require_carla()
    rclpy, Image, PointCloud2, PointField = _require_ros2()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    if args.town:
        world = client.load_world(args.town)
    else:
        world = client.get_world()

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise SystemExit("map has no spawn points")

    vehicle_bp = bp_lib.filter("vehicle.*")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True)

    cam_transform = carla.Transform(carla.Location(x=1.5, z=CAMERA_MOUNT_Z))

    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(IMG_W))
    rgb_bp.set_attribute("image_size_y", str(IMG_H))
    rgb_bp.set_attribute("fov", str(FOV))
    rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)

    sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(IMG_W))
    sem_bp.set_attribute("image_size_y", str(IMG_H))
    sem_bp.set_attribute("fov", str(FOV))
    sem_cam = world.spawn_actor(sem_bp, cam_transform, attach_to=vehicle)

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("range", "40")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=LIDAR_MOUNT_Z))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    rclpy.init()
    node = rclpy.create_node("carla_feed")
    rgb_pub = node.create_publisher(Image, "/camera/front/image", 5)
    lidar_pub = node.create_publisher(PointCloud2, "/lidar/points", 5)

    state = {"rgb": None, "sem": None, "lidar": None, "tick": 0}
    os.makedirs(args.dump_dir, exist_ok=True) if args.dump_dir else None

    def on_rgb(image):
        bgr = bgra_bytes_to_bgr(image.raw_data, image.height, image.width)
        state["rgb"] = bgr
        msg = Image()
        msg.header.frame_id = "camera_front"
        msg.height, msg.width = image.height, image.width
        msg.encoding = "bgr8"
        msg.step = image.width * 3
        msg.data = bgr.tobytes()
        rgb_pub.publish(msg)

    def on_sem(image):
        sem_bgr = bgra_bytes_to_bgr(image.raw_data, image.height, image.width)
        state["sem"] = sem_bgr
        # first-frame diagnostic: print observed tag ids so a wrong default
        # in carla_convert.semantic_to_road_mask(road_tags=...) is caught
        # immediately rather than silently producing a wrong IoU.
        if state["tick"] == 0:
            uniq = np.unique(sem_bgr[:, :, 2])
            node.get_logger().info(f"semantic tags observed this frame: {uniq.tolist()}")

    def on_lidar(measurement):
        pts = carla_lidar_to_rep103(measurement.raw_data, sensor_z=LIDAR_MOUNT_Z)
        state["lidar"] = pts
        stamp = node.get_clock().now().to_msg()
        lidar_pub.publish(build_pointcloud2(pts, "lidar_front", stamp, PointCloud2, PointField))

        state["tick"] += 1
        if args.dump_dir and state["rgb"] is not None and state["sem"] is not None \
                and state["tick"] % args.dump_every == 0:
            import cv2
            i = state["tick"]
            cv2.imwrite(os.path.join(args.dump_dir, f"rgb_{i:06d}.png"), state["rgb"])
            cv2.imwrite(os.path.join(args.dump_dir, f"sem_{i:06d}.png"), state["sem"])

    rgb_cam.listen(on_rgb)
    sem_cam.listen(on_sem)
    lidar.listen(on_lidar)

    print(f"[carla_feed] publishing /camera/front/image + /lidar/points "
         f"(BEST_EFFORT-compatible plain publishers). Ctrl-C to stop.")
    try:
        n = 0
        while args.ticks == 0 or n < args.ticks:
            world.wait_for_tick()
            rclpy.spin_once(node, timeout_sec=0.0)
            n += 1
    except KeyboardInterrupt:
        pass
    finally:
        for actor in (rgb_cam, sem_cam, lidar, vehicle):
            try:
                actor.destroy()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
