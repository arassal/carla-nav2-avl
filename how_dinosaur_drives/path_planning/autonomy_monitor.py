"""autonomy_monitor — source-agnostic IGVC §I.2 safety-light trigger.

Publishes /autonomous_mode (std_msgs/Bool, latched) = True whenever Nav2 is
actively executing a NavigateToPose goal, False otherwise. Unlike the old
mission_manager-only trigger, this watches the action SERVER status, so it
fires for *any* goal source — mission_manager, RViz "Nav2 Goal", a CLI
`ros2 action send_goal`, etc. — and stays True through recovery behaviors and
obstacle-pauses (the goal remains EXECUTING the whole time).

Why a separate always-on node (not folded into mission_manager): mission_manager
is gated off by default and only knows about its own goals, sent in the map
frame. The field-proven workflow sends goals manually in the odom frame, which
mission_manager never sees — so the light would stay SOLID while the robot
drives itself (a §I.4 qualification failure). This node closes that gap by
deriving autonomy from the pipeline's real state.

Contract:
  in : /navigate_to_pose/_action/status  (action_msgs/GoalStatusArray)
  out: /autonomous_mode                   (std_msgs/Bool, latched)
actuator_node streams A1/A0 to the Teensy from this flag (e-stop forces A0).
The webui AUTONOMOUS toggle remains an independent manual publisher on the same
topic for non-Nav2 / bench autonomy.
"""
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool


# Latched (transient-local, reliable) QoS for /autonomous_mode — MUST match the
# subscriber in actuator_node so a (re)started actuator picks up the current
# state immediately. Same profile mission_manager + webui_node use.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

# The action status topic is published by the server with reliable +
# transient_local + keep_last(1) (rcl_action's status-default profile). Match it
# or DDS reports incompatible QoS and we silently get nothing — plus the
# transient_local latch hands us the last status immediately on join.
ACTION_STATUS_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

# Goal states that mean "the robot is driving itself right now". CANCELING is
# included: a goal being canceled is still decelerating under Nav2 control, so
# it is still autonomous until it actually stops (then the goal goes terminal).
ACTIVE_STATES = frozenset((
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
    GoalStatus.STATUS_CANCELING,
))


class AutonomyMonitor(Node):
    def __init__(self) -> None:
        super().__init__('autonomy_monitor')

        # How long to keep the light flashing after the last active goal ends,
        # before reverting to SOLID. Bridges the brief no-active-goal gaps
        # between sequential waypoints and between a recovery finishing and the
        # next plan starting, so the light never blips solid mid-mission.
        self.declare_parameter('release_grace_s', 1.5)
        self.declare_parameter(
            'action_status_topic', '/navigate_to_pose/_action/status'
        )
        self.declare_parameter('eval_rate_hz', 5.0)

        self._grace = float(self.get_parameter('release_grace_s').value)
        status_topic = self.get_parameter('action_status_topic').value
        eval_rate = float(self.get_parameter('eval_rate_hz').value)

        self._any_active = False
        self._last_active_t = self.get_clock().now()
        self._published = None  # last Bool actually published

        self._auto_pub = self.create_publisher(
            Bool, '/autonomous_mode', LATCHED_QOS
        )
        # Establish a definite SOLID baseline on the latched topic at startup so
        # a subscriber that joins before the first goal sees a real value.
        self._set_autonomous(False)

        self.create_subscription(
            GoalStatusArray, status_topic, self._on_status, ACTION_STATUS_QOS
        )
        self.create_timer(1.0 / eval_rate, self._on_tick)

        self.get_logger().info(
            f'autonomy_monitor started — watching {status_topic}, '
            f'release_grace_s={self._grace}'
        )

    def _on_status(self, msg: GoalStatusArray) -> None:
        self._any_active = any(
            g.status in ACTIVE_STATES for g in msg.status_list
        )
        if self._any_active:
            self._last_active_t = self.get_clock().now()
            # Flash the instant a goal goes active — don't wait for the tick.
            self._set_autonomous(True)

    def _on_tick(self) -> None:
        if self._any_active:
            self._set_autonomous(True)
            return
        # No active goal — hold the flash until the grace window expires, then
        # drop to solid.
        elapsed = (self.get_clock().now() - self._last_active_t).nanoseconds * 1e-9
        if elapsed >= self._grace:
            self._set_autonomous(False)

    def _set_autonomous(self, val: bool) -> None:
        """Publish the §I.2 flag, but only on an actual change."""
        if val == self._published:
            return
        self._published = val
        self._auto_pub.publish(Bool(data=val))
        self.get_logger().info(
            f'autonomous_mode -> {val} (safety light {"FLASH" if val else "SOLID"})'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutonomyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
