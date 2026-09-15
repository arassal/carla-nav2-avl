import pytest

from perception_costmap import launch_guard as g


def test_camera_list_parses_and_keeps_order():
    assert g.parse_camera_list("right, front,,front") == ["right", "front"]
    assert g.parse_camera_list("") == []


def test_unknown_camera_is_rejected_by_name():
    with pytest.raises(ValueError, match="back"):
        g.parse_camera_list("front,back")


def test_finds_cameras_started_by_the_boot_script():
    boot = ("/usr/bin/python3 /opt/ros/humble/bin/ros2 launch zed_wrapper "
            "zed_camera.launch.py camera_model:=zedx camera_name:=zed_left "
            "serial_number:=49910017 publish_tf:=false")
    assert g.cameras_in_cmdlines([boot]) == {"left"}


def test_finds_cameras_by_their_component_container():
    container = ("/opt/ros/humble/lib/rclcpp_components/component_container_isolated "
                 "--use_multi_threaded_executor --ros-args -r __node:=zed_container "
                 "-r __ns:=/zed_front")
    assert g.cameras_in_cmdlines([container]) == {"front"}


def test_unrelated_processes_are_not_cameras():
    others = [
        "python3 -m perception_costmap.costmap_node --ros-args -r __ns:=/zed_front",
        "vim notes_camera_name:=zed_right.txt",
        "component_container --ros-args -r __ns:=/nav2",
    ]
    assert g.cameras_in_cmdlines(others) == set()


def test_reads_real_proc_cmdlines(tmp_path):
    (tmp_path / "123").mkdir()
    (tmp_path / "123" / "cmdline").write_bytes(
        b"ros2\0launch\0zed_wrapper\0zed_camera.launch.py\0camera_name:=zed_right\0")
    (tmp_path / "self").mkdir()  # non-numeric entries are ignored
    assert g.cameras_in_cmdlines(g.local_cmdlines(str(tmp_path))) == {"right"}


def test_missing_proc_is_empty_not_an_error(tmp_path):
    assert g.local_cmdlines(str(tmp_path / "nope")) == []


def test_cameras_on_graph():
    nodes = {("/zed_front", "zed_node"), ("/", "perception_costmap"),
             ("/zed_left", "zed_container"), ("/", "robot_state_publisher")}
    assert g.cameras_on_graph(nodes) == {"front"}


def test_plan_staggers_cameras_after_sensors_settle():
    steps = g.plan(["front", "left", "right"], set(), True, True,
                   camera_delay=40, sensors_settle=15)
    assert steps["cameras"] == [("front", 15.0), ("left", 55.0), ("right", 95.0)]
    assert steps["skipped"] == [] and steps["sensors"] and steps["costmap"]


def test_plan_skips_running_cameras_without_leaving_a_gap():
    """Someone already has the front camera up: start left immediately."""
    steps = g.plan(["front", "left", "right"], {"front"}, False, True,
                   camera_delay=40, sensors_settle=15)
    assert steps["cameras"] == [("left", 0.0), ("right", 40.0)]
    assert steps["skipped"] == ["front"]


def test_plan_with_everything_running_starts_no_cameras():
    steps = g.plan(["front"], {"front", "left"}, False, False,
                   camera_delay=40, sensors_settle=15)
    assert steps["cameras"] == [] and steps["skipped"] == ["front"]
