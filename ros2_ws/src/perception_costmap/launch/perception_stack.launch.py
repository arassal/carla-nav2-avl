"""
Bring up only what perception needs on the car: sensors (TF), the ZED
cameras you ask for with lean settings, and the costmap node.

    # whole perception stack: sensors + front/left/right + costmap
    ros2 launch perception_costmap perception_stack.launch.py

    # just the front camera and the costmap (e.g. tuning the front view)
    ros2 launch perception_costmap perception_stack.launch.py cameras:=front

    # only a camera, nothing else
    ros2 launch perception_costmap perception_stack.launch.py \
        cameras:=left sensors:=false costmap:=false

What it deliberately does NOT start, compared with
deploy/full_stack_restart.sh: EKF, viz_node, costmap_rgb, web_video_server,
the dashboard server and RViz. Run those separately when you want to watch.

Cameras use config/zed_perception_<name>.yaml: RGB, depth and the confidence
map only -- no point cloud, no positional tracking, no IMU, processing capped
at 8 frames/s. Before any of that, it checks for things already running
(ROS graph + this machine's processes) and reuses them instead of starting a
second copy -- a second driver on a live ZED X wedges the GMSL link.
Details and the checks themselves: perception_costmap/launch_guard.py.

ROS_DOMAIN_ID: the boot service runs on domain 0 but login shells on the car
default to 42. On the car (avros_bringup installed) this launch pins domain 0
unless you pass ros_domain_id:=N, so it sees and reuses the boot stack.
"""

import os
import time

from ament_index_python.packages import (
    PackageNotFoundError, get_package_share_directory)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction,
    SetEnvironmentVariable, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from perception_costmap import launch_guard

# Serial per camera, as run by deploy/full_stack_restart.sh (the boot path the
# 2026-07-09 calibration was done on). avros_bringup's sensors.launch.py has
# left and right SWAPPED relative to this -- see ISSUES.md C8.
SERIALS = {"front": "42569280", "left": "49910017", "right": "43779087"}


def _share(package):
    try:
        return get_package_share_directory(package)
    except PackageNotFoundError:
        return None


def _running_on_graph(wait_sec):
    """(namespace, name) of every node visible on the current domain."""
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    context = rclpy.context.Context()
    # A throwaway probe inside the launch process: leave signal handling to
    # launch rather than letting rclpy install process-wide handlers.
    rclpy.init(context=context, signal_handler_options=SignalHandlerOptions.NO)
    try:
        probe = rclpy.create_node("perception_stack_probe", context=context)
        time.sleep(wait_sec)  # let discovery find the existing graph
        nodes = {(ns, name) for name, ns in probe.get_node_names_and_namespaces()}
        probe.destroy_node()
    finally:
        rclpy.shutdown(context=context)
    return nodes


def _bool(context, name):
    return context.launch_configurations[name].strip().lower() in ("1", "true", "yes")


def _setup(context):
    cfg = context.launch_configurations
    actions = []
    avros = _share("avros_bringup")

    # Environment first: the graph probe below must use the same domain and
    # middleware as the stack it is looking for.
    domain = cfg["ros_domain_id"].strip() or ("0" if avros else "")
    if domain:
        os.environ["ROS_DOMAIN_ID"] = domain
        actions.append(SetEnvironmentVariable("ROS_DOMAIN_ID", domain))
    if avros and _share("rmw_cyclonedds_cpp"):
        cyclone = "file://" + os.path.join(avros, "config", "cyclonedds.xml")
        for key, value in (("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
                           ("CYCLONEDDS_URI", cyclone)):
            os.environ[key] = value
            actions.append(SetEnvironmentVariable(key, value))

    requested = launch_guard.parse_camera_list(cfg["cameras"])
    want_sensors, want_costmap = _bool(context, "sensors"), _bool(context, "costmap")

    graph = set()
    running = set()
    if _bool(context, "reuse_running"):
        graph = _running_on_graph(float(cfg["discovery_wait"]))
        running = (launch_guard.cameras_on_graph(graph)
                   | launch_guard.cameras_in_cmdlines(launch_guard.local_cmdlines()))
        if want_sensors and ("/", "robot_state_publisher") in graph:
            actions.append(LogInfo(msg="[perception_stack] sensors already running -- reusing"))
            want_sensors = False
        if want_costmap and ("/", "perception_costmap") in graph:
            actions.append(LogInfo(msg="[perception_stack] costmap node already running -- reusing"))
            want_costmap = False

    steps = launch_guard.plan(
        requested, running, want_sensors, want_costmap,
        camera_delay=float(cfg["camera_start_delay"]),
        sensors_settle=float(cfg["sensors_settle"]))

    actions.append(LogInfo(msg="[perception_stack] domain=%s cameras=%s already running=%s "
                               "start sensors=%s costmap=%s" % (
                                   os.environ.get("ROS_DOMAIN_ID", "(default 0)"),
                                   [c for c, _ in steps["cameras"]], steps["skipped"],
                                   steps["sensors"], steps["costmap"])))

    if steps["sensors"]:
        if not avros:
            raise RuntimeError("sensors:=true needs avros_bringup (github.com/Paarseus/IGVC_ROS2); "
                               "source its install, or pass sensors:=false")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(avros, "launch", "sensors.launch.py")),
            launch_arguments={
                "enable_velodyne": cfg["lidar"],
                # cameras are started below, one at a time, with lean settings
                "enable_zed_front": "false",
                "enable_zed_left": "false",
                "enable_zed_right": "false",
            }.items()))

    if steps["cameras"]:
        zed = _share("zed_wrapper")
        if not zed:
            raise RuntimeError("cameras:=%s needs zed_wrapper; source its install, or pass cameras:=''"
                               % cfg["cameras"])
        here = get_package_share_directory("perception_costmap")
        for name, start_after in steps["cameras"]:
            camera = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(zed, "launch", "zed_camera.launch.py")),
                launch_arguments={
                    "camera_model": "zedx",
                    "camera_name": "zed_" + name,
                    "serial_number": cfg[name + "_serial"],
                    "publish_tf": "false",     # robot_localization owns odom -> base_link
                    "publish_urdf": "false",   # the avros URDF already includes the cameras
                    "ros_params_override_path": os.path.join(
                        here, "config", "zed_perception_%s.yaml" % name),
                }.items())
            actions.append(LogInfo(msg="[perception_stack] zed_%s starts in %.0f s"
                                       % (name, start_after)))
            actions.append(TimerAction(period=start_after, actions=[camera])
                           if start_after > 0 else camera)

    if steps["costmap"]:
        actions.append(Node(
            package="perception_costmap", executable="costmap_node",
            name="perception_costmap", output="screen",
            parameters=[cfg["config"]]))

    return actions


def generate_launch_description():
    share = get_package_share_directory("perception_costmap")
    args = [
        DeclareLaunchArgument("cameras", default_value="front,left,right",
                              description="Comma-separated cameras to run: front,left,right. "
                                          "'none' for sensors and costmap only."),
        DeclareLaunchArgument("sensors", default_value="true",
                              description="Start avros_bringup sensors.launch.py (URDF TF, Xsens, Velodyne). "
                                          "The ZED driver waits for this TF."),
        DeclareLaunchArgument("lidar", default_value="true",
                              description="Velodyne. perception_dinosaur.yaml does not use it, "
                                          "but Nav2 (auto_drive) does."),
        DeclareLaunchArgument("costmap", default_value="true",
                              description="Start the perception_costmap node."),
        DeclareLaunchArgument("config", default_value=os.path.join(
                                  share, "config", "perception_dinosaur.yaml"),
                              description="perception_costmap params YAML."),
        DeclareLaunchArgument("reuse_running", default_value="true",
                              description="Skip anything already running instead of starting a second copy."),
        DeclareLaunchArgument("ros_domain_id", default_value="",
                              description="Empty: 0 on the car (matches the boot service), "
                                          "otherwise inherit the shell."),
        DeclareLaunchArgument("camera_start_delay", default_value="40.0",
                              description="Seconds between camera starts. Parallel starts wedge GMSL."),
        DeclareLaunchArgument("sensors_settle", default_value="15.0",
                              description="Seconds to wait after starting sensors before the first camera."),
        DeclareLaunchArgument("discovery_wait", default_value="2.0",
                              description="Seconds to listen for already-running nodes."),
    ] + [
        DeclareLaunchArgument(name + "_serial", default_value=serial,
                              description="ZED X serial number for the %s camera." % name)
        for name, serial in SERIALS.items()
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
