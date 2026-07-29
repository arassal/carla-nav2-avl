#!/usr/bin/env python3
"""
CARLA 0.9.16 -> ROS2 feed for perception_costmap. Runs on the x86/5090 box
(CARLA has no ARM build; the Jetson never sees this file).

Spawns an autopilot ego with front RGB + semantic camera + lidar and
publishes:
    /camera/front/image        sensor_msgs/Image (bgr8)
    /camera/front/camera_info  sensor_msgs/CameraInfo (K from FOV)
    /lidar/points              sensor_msgs/PointCloud2 (REP-103, base_link)

--dump-dir writes paired frames (NNNN_rgb.png / NNNN_sem.png) every
--dump-every ticks for tools/eval_road_iou.py.

    python3 tools/carla_feed.py --host 127.0.0.1 --dump-dir /tmp/pairs
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.carla_convert import (
    bgra_bytes_to_bgr, carla_lidar_to_rep103)

W, H, FOV = 640, 360, 90.0
CAM_X, CAM_Z, CAM_PITCH = 1.5, 1.6, -10.0     # CARLA pitch: negative = down
LIDAR_Z = 1.8


def make_camera_info(stamp, frame):
    from sensor_msgs.msg import CameraInfo
    fx = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
    msg = CameraInfo()
    msg.header.stamp, msg.header.frame_id = stamp, frame
    msg.width, msg.height = W, H
    msg.k = [fx, 0.0, W / 2.0, 0.0, fx, H / 2.0, 0.0, 0.0, 1.0]
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--dump-every", type=int, default=20)
    args = ap.parse_args()

    import carla
    import cv2
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    rclpy.init()
    node = rclpy.create_node("carla_feed")
    pub_img = node.create_publisher(Image, "/camera/front/image", qos_profile_sensor_data)
    pub_info = node.create_publisher(CameraInfo, "/camera/front/camera_info", qos_profile_sensor_data)
    pub_pts = node.create_publisher(PointCloud2, "/lidar/points", qos_profile_sensor_data)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp = world.get_blueprint_library()

    ego = world.spawn_actor(bp.filter("vehicle.tesla.model3")[0],
                            world.get_map().get_spawn_points()[0])
    ego.set_autopilot(True)

    def cam_bp(kind):
        b = bp.find(kind)
        b.set_attribute("image_size_x", str(W))
        b.set_attribute("image_size_y", str(H))
        b.set_attribute("fov", str(FOV))
        return b

    cam_tf = carla.Transform(carla.Location(x=CAM_X, z=CAM_Z),
                             carla.Rotation(pitch=CAM_PITCH))
    rgb = world.spawn_actor(cam_bp("sensor.camera.rgb"), cam_tf, attach_to=ego)
    sem = world.spawn_actor(cam_bp("sensor.camera.semantic_segmentation"),
                            cam_tf, attach_to=ego)
    lidar_bp = bp.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "30")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar = world.spawn_actor(lidar_bp,
                              carla.Transform(carla.Location(z=LIDAR_Z)),
                              attach_to=ego)

    state = {"rgb": None, "sem": None, "n": 0}
    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    def now():
        return node.get_clock().now().to_msg()

    def on_rgb(image):
        img = bgra_bytes_to_bgr(bytes(image.raw_data), H, W)
        state["rgb"] = img
        stamp = now()
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, "camera_front"
        msg.height, msg.width = H, W
        msg.encoding, msg.step = "bgr8", W * 3
        msg.data = img.tobytes()
        pub_img.publish(msg)
        pub_info.publish(make_camera_info(stamp, "camera_front"))

    def on_sem(image):
        state["sem"] = bgra_bytes_to_bgr(bytes(image.raw_data), H, W)
        state["n"] += 1
        if dump and state["rgb"] is not None and state["n"] % args.dump_every == 0:
            i = state["n"] // args.dump_every
            cv2.imwrite(str(dump / ("%04d_rgb.png" % i)), state["rgb"])
            cv2.imwrite(str(dump / ("%04d_sem.png" % i)), state["sem"])
            tags = sorted(np.unique(state["sem"][:, :, 2]).tolist())
            print("pair %04d saved; semantic tags present: %s" % (i, tags))

    def on_lidar(meas):
        pts = carla_lidar_to_rep103(bytes(meas.raw_data), LIDAR_Z)
        hdr = Header()
        hdr.stamp, hdr.frame_id = now(), "base_link"
        pub_pts.publish(point_cloud2.create_cloud_xyz32(hdr, pts.tolist()))

    rgb.listen(on_rgb)
    sem.listen(on_sem)
    lidar.listen(on_lidar)
    print("feeding ROS2. IPM 'camera' mode params for the YAML: cam_x=%.2f cam_height=%.2f cam_pitch_deg=%.1f. Ctrl-C to stop." % (CAM_X, CAM_Z, -CAM_PITCH))
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for s in (rgb, sem, lidar):
            s.stop()
        for a in (rgb, sem, lidar, ego):
            a.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
