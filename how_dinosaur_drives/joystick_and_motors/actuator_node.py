"""
Actuator bridge node: cmd_vel / ActuatorCommand -> Teensy serial -> SparkMAX.

Diff-drive kinematics for an AndyMark Raptor TRACKED chassis:
    motor RPM -> ground m/s via 12.75:1 gearbox + 20T drive pulley
    commanded (linear, angular) -> (L_rpm, R_rpm) via diff-drive inverse

Skid-steer correction (Mandow ICR model): tracked vehicles can't pure-roll
when rotating — both tracks have to scrape sideways against the ground.
The standard `diff_drive_controller` accounts for this with a single multiplier
on the wheel separation (Husky uses 1.875, Jackal 1.5; we use 1.19 from
measured α = 0.84). See `wheel_separation_multiplier` param and references:
    - ros2_controllers/diff_drive_controller (humble) — same convention
    - Mandow et al. 2007 IROS "Experimental kinematics for wheeled skid-steer
      mobile robots" doi:10.1109/IROS.2007.4399139
    - Clearpath Jackal control.yaml `wheel_separation_multiplier: 1.5`

Heading-hold: when commanded angular velocity is near zero, IMU yaw is
locked and a proportional correction is applied to ω. Useful for teleop
on uneven ground; Nav2 RPP/MPPI handles this via path-tracking when navigating.

Note: actuator-level IMU yaw_rate feedback was REMOVED 2026-05-18 — it was
non-standard (Macenski / Nav2 #5524: closed-loop at the actuator is only for
"highly delayed or particularly low-quality odometry"). The standard layering
fuses IMU vyaw in robot_localization EKF, and MPPI reads /odometry/filtered.

Subscribes:
  /cmd_vel                  geometry_msgs/Twist      (Nav2, teleop)
  /avros/actuator_command   avros_msgs/ActuatorCommand (webui direct)
  /imu/data                 sensor_msgs/Imu          (Xsens yaw for heading-hold)

Publishes:
  /avros/actuator_state     avros_msgs/ActuatorState @ 20 Hz
  /wheel_odom               nav_msgs/Odometry @ 50 Hz (for EKF fusion)
"""

import math
import threading
import time
from typing import List

import rclpy
import serial
from action_msgs.msg import GoalStatus, GoalStatusArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension
from avros_msgs.msg import ActuatorCommand, ActuatorState

from .command_arbitration import select_command_source


# Latched (transient-local) QoS for the /autonomous_mode mode flag, so a
# late-joining or restarted actuator_node picks up the current mode without
# waiting for the next publish. Publishers (mission_manager, webui_node) must
# match. depth=1: only the latest mode matters.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

ACTION_STATUS_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

ACTIVE_NAV_STATES = frozenset((
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
    GoalStatus.STATUS_CANCELING,
))


# Runtime-tunable parameters: `ros2 param set /actuator_node <name> <value>`
# applies immediately. Mapping: `ros_param_name -> self attr`.
#
# Any param NOT listed here is init-only — `ros2 param set` on it is rejected
# with a clear reason so the caller knows to relaunch. Validation in
# `_on_param_change` keeps the values non-negative.
#
# Why the split:
#   - geometry / serial / frames: hardware-defining; runtime change is unsafe
#   - timer rates: create_timer is fixed at construction; needs relaunch
#   - clamps + slew + heading-hold + SparkMAX PID gains: pure software state
#     OR re-pushed to Teensy on change (PID gains)
#
# SparkMAX gains (kFF/kP/kI/kD/kIZone) are RAM-only on change — they sit in
# the controller's working set but are NOT persisted to SparkMAX flash. After
# a successful tuning session, `BURN` over Teensy serial (or REV Hardware
# Client) writes the working gains to flash so they survive a power-cycle.
_DYNAMIC_PARAMS = {
    'max_linear_mps':              '_max_v',
    'max_angular_rps':             '_max_w',
    'max_linear_accel_mps2':       '_accel_v',
    'max_linear_decel_mps2':       '_decel_v',
    'max_angular_accel_rps2':      '_accel_w',
    'heading_hold_deadband':       '_hh_deadband',
    'heading_kp':                  '_heading_kp',
    'wheel_separation_multiplier': '_ws_multiplier',
    'cmd_timeout_s':               '_cmd_timeout',
    'kFF':                         '_k_ff',
    'kP':                          '_k_p',
    'kI':                          '_k_i',
    'kD':                          '_k_d',
    'kIZone':                      '_k_izone',
}

# Which dynamic params need to be pushed to the Teensy on change. Keys here
# must also appear in _DYNAMIC_PARAMS; values are the single-letter prefix
# the Teensy protocol uses (firmware/teensy_diff_drive parses K[FPID Z]<val>).
_PID_SERIAL_PREFIX = {
    'kFF':    'KF',
    'kP':     'KP',
    'kI':     'KI',
    'kD':     'KD',
    'kIZone': 'KZ',
}


def yaw_from_quaternion(q) -> float:
    """Extract yaw (Z rotation) from a geometry_msgs.Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a: float) -> float:
    """Wrap radians to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class ActuatorNode(Node):
    """Diff-drive bridge: ROS2 commands <-> Teensy serial protocol."""

    def __init__(self):
        super().__init__('actuator_node')

        # ---- parameters ----
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('serial_baud', 115200)
        self.declare_parameter('track_width_m', 0.7366)       # 29 inches
        # Raptor + TBMini 12.75:1 gearbox and 20-tooth drive pulley.
        self.declare_parameter('m_per_motor_rev', 0.01994)
        self.declare_parameter('max_linear_mps', 1.5)
        self.declare_parameter('max_angular_rps', 1.0)
        # Slew-rate limiting: smooths velocity changes to protect both the
        # 12V rail (motor inrush causes Jetson brown-outs) and passengers
        # (decel is felt as a jolt). Applied to the target (v, ω) before
        # heading-hold corrections and diff-drive inverse kinematics.
        self.declare_parameter('max_linear_accel_mps2', 1.0)
        self.declare_parameter('max_linear_decel_mps2', 1.5)
        self.declare_parameter('max_angular_accel_rps2', 2.0)
        self.declare_parameter('heading_hold_deadband', 0.05)  # rad/s
        self.declare_parameter('heading_kp', 1.5)             # heading-hold P gain
        # Mandow skid-steer correction: effective wheel separation =
        # track_width_m * wheel_separation_multiplier. Default 1.0 = no
        # correction (pure-rolling assumption). Tracked chassis on our
        # surface measured α=0.84 -> multiplier=1.19. Same convention as
        # ros2_controllers/diff_drive_controller; Husky=1.875, Jackal=1.5.
        self.declare_parameter('wheel_separation_multiplier', 1.0)
        self.declare_parameter('cmd_timeout_s', 0.5)
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('state_pub_rate_hz', 20.0)
        # SparkMAX PID gains to set on startup (from Phase 6 tuning)
        self.declare_parameter('kFF', 0.000197)
        self.declare_parameter('kP', 0.0004)
        self.declare_parameter('kI', 0.0)
        self.declare_parameter('kD', 0.0)
        self.declare_parameter('kIZone', 200.0)
        # Odom frame names
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        p = self.get_parameter
        self._port = p('serial_port').value
        self._baud = p('serial_baud').value
        self._track_w = p('track_width_m').value
        self._m_per_rev = p('m_per_motor_rev').value
        self._max_v = p('max_linear_mps').value
        self._max_w = p('max_angular_rps').value
        self._accel_v = p('max_linear_accel_mps2').value
        self._decel_v = p('max_linear_decel_mps2').value
        self._accel_w = p('max_angular_accel_rps2').value
        self._hh_deadband = p('heading_hold_deadband').value
        self._heading_kp = p('heading_kp').value
        self._ws_multiplier = p('wheel_separation_multiplier').value
        self._cmd_timeout = p('cmd_timeout_s').value
        self._k_ff = p('kFF').value
        self._k_p = p('kP').value
        self._k_i = p('kI').value
        self._k_d = p('kD').value
        self._k_izone = p('kIZone').value
        self._odom_frame = p('odom_frame').value
        self._base_frame = p('base_frame').value

        control_rate = p('control_rate_hz').value
        state_rate = p('state_pub_rate_hz').value

        # ---- serial ----
        self._serial = serial.Serial(self._port, self._baud, timeout=0.1)
        time.sleep(0.3)
        self._serial.reset_input_buffer()
        self._serial_lock = threading.Lock()
        self.get_logger().info(f'Serial open: {self._port} @ {self._baud}')

        # Push PID gains to the Teensy (which forwards to both SparkMAXes via
        # PARAMETER_WRITE cls=14). Also re-pushed on dynamic-param change —
        # see _on_param_change.
        for name, val in [('KF', self._k_ff), ('KP', self._k_p),
                          ('KI', self._k_i), ('KD', self._k_d),
                          ('KZ', self._k_izone)]:
            self._serial_write(f'{name}{val}')
            time.sleep(0.2)
        self.get_logger().info(
            f'SparkMAX gains set: kFF={self._k_ff} kP={self._k_p} '
            f'kI={self._k_i} kD={self._k_d} kIZone={self._k_izone}'
        )

        # ---- state ----
        # Command targets (set by callbacks; read by control loop)
        # Keep source targets separate. Sharing these values made a centered
        # 20 Hz Web UI overwrite Nav2's cmd_vel even while a goal was active.
        self._cmd_vel_v = 0.0
        self._cmd_vel_w = 0.0
        self._actuator_v = 0.0
        self._actuator_w = 0.0
        # Slewed values — the actually-commanded setpoints, rate-limited
        # toward the targets in the control loop.
        self._slew_v = 0.0
        self._slew_w = 0.0
        self._estop = False
        self._last_cmd_vel_t = None           # rclpy Time or None
        self._last_actuator_cmd_t = None

        # IGVC §I.2 safety-light state. The light flashes only when the vehicle
        # is genuinely autonomous; e-stop forces it solid. The (A1/A0) command
        # is streamed to the Teensy every control cycle as the light heartbeat
        # (firmware reverts to SOLID if the stream stops — fail-safe).
        self._autonomous_requested = False
        self._nav_active = False
        self._last_light_sent = None          # 'A1' / 'A0' actually streamed

        # Heading-hold state
        self._heading_locked = False
        self._heading_target = 0.0
        self._current_yaw = 0.0
        self._current_yaw_rate = 0.0
        self._imu_fresh = False

        # Feedback state (from Teensy E-lines)
        self._l_meas_rpm = 0.0
        self._r_meas_rpm = 0.0
        self._l_meas_pos = 0.0    # motor revolutions, cumulative signed
        self._r_meas_pos = 0.0
        self._l_pos_prev = None
        self._r_pos_prev = None
        self._fb_lock = threading.Lock()

        # Integrated odometry (pose from wheel encoders)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._odom_last_t = self.get_clock().now()

        # ---- subscribers ----
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(
            ActuatorCommand, '/avros/actuator_command',
            self._on_actuator_cmd, 10
        )
        self.create_subscription(Imu, '/imu/data', self._on_imu, 20)
        # Safety-light autonomy flag — published by mission_manager (auto) and/or
        # the webui AUTONOMOUS toggle (manual). Latched so we get the current
        # value immediately on (re)start.
        self.create_subscription(
            Bool, '/autonomous_mode', self._on_autonomous, LATCHED_QOS
        )
        self.create_subscription(
            GoalStatusArray, '/navigate_to_pose/_action/status',
            self._on_nav_status, ACTION_STATUS_QOS
        )

        # ---- publishers ----
        self._state_pub = self.create_publisher(
            ActuatorState, '/avros/actuator_state', 10
        )
        self._odom_pub = self.create_publisher(Odometry, '/wheel_odom', 10)
        # Raw per-wheel telemetry for offline motor-sync analysis.
        # Layout (16 floats): see _control_loop end-of-tick publish.
        self._wheel_debug_pub = self.create_publisher(
            Float32MultiArray, '/avros/wheel_debug', 50
        )
        self._wheel_debug_labels = [
            'L_cmd_rpm', 'R_cmd_rpm',
            'L_meas_rpm', 'R_meas_rpm',
            'L_pos_rev', 'R_pos_rev',
            'v_target', 'w_target',
            'v_slewed', 'w_slewed',
            'v_after_imu', 'w_after_imu',
            'yaw', 'yaw_rate',
            'heading_locked', 'estop',
        ]

        # Diagonal covariance for 2D diff-drive wheel odometry.
        # Trust velocity (wheels measure it directly), mark unobservable
        # axes (z, roll, pitch) with effectively-infinite covariance so
        # robot_localization ignores them.
        self._pose_cov = [0.0] * 36
        self._pose_cov[0] = 0.001     # x
        self._pose_cov[7] = 0.001     # y
        self._pose_cov[14] = 1e6      # z
        self._pose_cov[21] = 1e6      # roll
        self._pose_cov[28] = 1e6      # pitch
        self._pose_cov[35] = 0.01     # yaw
        self._twist_cov = [0.0] * 36
        self._twist_cov[0] = 0.0001   # vx
        self._twist_cov[7] = 1e6      # vy (no lateral slip on diff drive)
        self._twist_cov[14] = 1e6     # vz
        self._twist_cov[21] = 1e6     # vroll
        self._twist_cov[28] = 1e6     # vpitch
        self._twist_cov[35] = 0.0001  # vyaw

        # ---- threads / timers ----
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._serial_reader, daemon=True
        )
        self._reader_thread.start()

        self._ctrl_dt = 1.0 / control_rate
        self.create_timer(self._ctrl_dt, self._control_loop)
        self.create_timer(1.0 / state_rate, self._publish_state)

        # Runtime parameter callback — see _DYNAMIC_PARAMS at module top.
        # Registered AFTER initial parameter reads so launch-time yaml
        # overrides don't trip the validator.
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info('Actuator node ready — diff-drive with heading-hold')

    # ------------------------------------------------------------- callbacks
    def _on_cmd_vel(self, msg: Twist):
        self._cmd_vel_v = max(-self._max_v, min(self._max_v, msg.linear.x))
        self._cmd_vel_w = max(-self._max_w, min(self._max_w, msg.angular.z))
        self._last_cmd_vel_t = self.get_clock().now()

    def _on_actuator_cmd(self, msg: ActuatorCommand):
        self._last_actuator_cmd_t = self.get_clock().now()
        if msg.estop:
            self._estop = True
            self._actuator_v = 0.0
            self._actuator_w = 0.0
            self.get_logger().warn('E-STOP via actuator_command')
            return
        self._estop = False
        # Map webui throttle/brake/steer -> (v, ω). steer follows REP-103
        # (CCW / left = +), same convention as angular.z.
        v = (msg.throttle - msg.brake) * self._max_v
        w = msg.steer * self._max_w
        self._actuator_v = max(-self._max_v, min(self._max_v, v))
        self._actuator_w = max(-self._max_w, min(self._max_w, w))

    def _on_imu(self, msg: Imu):
        self._current_yaw = yaw_from_quaternion(msg.orientation)
        self._current_yaw_rate = msg.angular_velocity.z
        self._imu_fresh = True

    def _on_autonomous(self, msg: Bool):
        """Latch the requested autonomy state for the §I.2 safety light."""
        self._autonomous_requested = bool(msg.data)

    def _on_nav_status(self, msg: GoalStatusArray):
        """Track actual Nav2 execution for unambiguous command arbitration."""
        self._nav_active = any(
            goal.status in ACTIVE_NAV_STATES for goal in msg.status_list
        )

    def _on_param_change(self, params: List[Parameter]) -> SetParametersResult:
        """
        Apply runtime parameter updates to the live node.

        Two-pass validate-then-apply so a partial mutation never lands:
        if any param in the batch is rejected, NONE are applied.
        """
        for p in params:
            if p.name not in _DYNAMIC_PARAMS:
                return SetParametersResult(
                    successful=False,
                    reason=f'{p.name} is init-only — relaunch actuator_node to change',
                )
            try:
                val = float(p.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False,
                    reason=f'{p.name}: expected a number, got {p.value!r}',
                )
            if val < 0.0:
                return SetParametersResult(
                    successful=False,
                    reason=f'{p.name} must be non-negative, got {val}',
                )
        for p in params:
            setattr(self, _DYNAMIC_PARAMS[p.name], float(p.value))
            if p.name in _PID_SERIAL_PREFIX:
                # Push to Teensy → SparkMAX RAM. Persistence (BURN to flash)
                # is a separate step the operator runs after a tuning session.
                line = f'{_PID_SERIAL_PREFIX[p.name]}{float(p.value)}'
                self._serial_write(line)
                self.get_logger().info(
                    f'  -> Teensy serial write: {line!r} (pushes to BOTH SparkMAXes)'
                )
            self.get_logger().info(
                f'param updated: {p.name} = {p.value} (live)'
            )
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------- control
    def _control_loop(self):
        now = self.get_clock().now()

        # Source-aware freshness. During an active Nav2 action, only cmd_vel is
        # eligible; direct Web UI commands still carry E-stop but cannot replace
        # the navigation target with a centered zero command.
        has_actuator = (
            self._last_actuator_cmd_t is not None
            and (now - self._last_actuator_cmd_t).nanoseconds / 1e9 < self._cmd_timeout
        )
        has_cmd_vel = (
            self._last_cmd_vel_t is not None
            and (now - self._last_cmd_vel_t).nanoseconds / 1e9 < self._cmd_timeout
        )

        source = select_command_source(
            self._estop, self._nav_active, has_actuator, has_cmd_vel)
        if source == 'actuator':
            v_req, w_req = self._actuator_v, self._actuator_w
        elif source == 'cmd_vel':
            v_req, w_req = self._cmd_vel_v, self._cmd_vel_w
        else:
            v_req, w_req = 0.0, 0.0

        # Slew-rate limit toward the requested target. Asymmetric accel/decel:
        # |slew| increasing toward request uses accel cap; decreasing uses
        # the (usually larger) decel cap. This smooths cmd_vel steps so
        # motor current draw is gradual — protects the 12V rail + passengers.
        if self._estop:
            # Emergency path: allow fastest decel to zero regardless of caps
            self._slew_v = 0.0
            self._slew_w = 0.0
        else:
            dv = v_req - self._slew_v
            dw = w_req - self._slew_w
            if abs(v_req) > abs(self._slew_v) and v_req * self._slew_v >= 0:
                max_dv = self._accel_v * self._ctrl_dt
            else:
                max_dv = self._decel_v * self._ctrl_dt
            max_dw = self._accel_w * self._ctrl_dt
            self._slew_v += max(-max_dv, min(max_dv, dv))
            self._slew_w += max(-max_dw, min(max_dw, dw))

        v_slewed = self._slew_v
        w_slewed = self._slew_w
        v = v_slewed
        w = w_slewed

        # IMU-based heading hold (P controller, straight-line intent only).
        # When |w_cmd| < deadband AND moving, lock yaw to current heading and
        # apply a P correction on yaw error. Released the moment a real ω is
        # commanded -- no actuator-level loop on ω. The IMU's yaw_rate is
        # fused into robot_localization EKF; Nav2 MPPI sees /odometry/filtered
        # and closes the ω loop at the controller layer (per Macenski, Nav2 #5524
        # and the Clearpath A300 / Husky / Jackal reference architecture).
        if self._imu_fresh:
            if abs(w) < self._hh_deadband and abs(v) > 0.02:
                if not self._heading_locked:
                    self._heading_target = self._current_yaw
                    self._heading_locked = True
                yaw_err = wrap_angle(self._heading_target - self._current_yaw)
                w = self._heading_kp * yaw_err
                w = max(-self._max_w * 0.5, min(self._max_w * 0.5, w))
            else:
                self._heading_locked = False

        # Diff-drive inverse kinematics with Mandow skid-steer correction.
        # Same pattern as ros2_controllers/diff_drive_controller — multiplier
        # widens the effective wheel separation to compensate for the lateral
        # track slip that violates the pure-rolling assumption.
        effective_separation = self._track_w * self._ws_multiplier
        l_mps = v - w * effective_separation / 2.0
        r_mps = v + w * effective_separation / 2.0
        l_rpm = l_mps / self._m_per_rev * 60.0
        r_rpm = r_mps / self._m_per_rev * 60.0

        # Send setpoint (or S on idle/estop)
        # Gate on the slewed setpoints (the actual inputs, which reach exactly 0
        # when the command goes stale) -- NOT the post-IMU v/w, whose w is kept
        # perpetually nonzero by the yaw-rate term acting on IMU noise. Using the
        # post-IMU w meant S (duty-0 -> brake-idle) never fired on the
        # cmd_vel-goes-stale path, leaving the motor creeping in active velocity
        # hold. The webui path dodged this because it stops via estop.
        sent_stop = self._estop or (
            source == 'stop'
            and abs(v_slewed) < 1e-6 and abs(w_slewed) < 1e-6
        )
        if sent_stop:
            self._serial_write('S')
            l_rpm_sent = 0.0
            r_rpm_sent = 0.0
        else:
            self._serial_write(f'L{l_rpm:.0f} R{r_rpm:.0f}')
            l_rpm_sent = l_rpm
            r_rpm_sent = r_rpm

        # IGVC §I.2 safety light: flash only when genuinely autonomous; e-stop
        # forces solid. Streamed every cycle as the Teensy light heartbeat (the
        # firmware reverts to SOLID if this stops). The Teensy acks only on a
        # real state change, so re-asserting the same line every cycle is quiet.
        light_line = (
            'A1' if self._autonomous_requested and not self._estop else 'A0'
        )
        self._serial_write(light_line)
        if light_line != self._last_light_sent:
            label = 'FLASH (autonomous)' if light_line == 'A1' else 'SOLID (manual)'
            self.get_logger().info(f'safety light -> {label}')
            self._last_light_sent = light_line

        # Publish raw per-wheel telemetry for offline analysis
        with self._fb_lock:
            l_meas = self._l_meas_rpm
            r_meas = self._r_meas_rpm
            l_pos = self._l_meas_pos
            r_pos = self._r_meas_pos
        dbg = Float32MultiArray()
        dbg.layout.dim.append(MultiArrayDimension(
            label=','.join(self._wheel_debug_labels),
            size=len(self._wheel_debug_labels),
            stride=len(self._wheel_debug_labels),
        ))
        dbg.data = [
            float(l_rpm_sent), float(r_rpm_sent),
            float(l_meas), float(r_meas),
            float(l_pos), float(r_pos),
            float(v_req), float(w_req),
            float(v_slewed), float(w_slewed),
            float(v), float(w),
            float(self._current_yaw), float(self._current_yaw_rate),
            1.0 if self._heading_locked else 0.0,
            1.0 if self._estop else 0.0,
        ]
        self._wheel_debug_pub.publish(dbg)

    # ------------------------------------------------------------- publishing
    def _publish_state(self):
        now = self.get_clock().now()
        with self._fb_lock:
            l_rpm = self._l_meas_rpm
            r_rpm = self._r_meas_rpm

        # Legacy ActuatorState contract (for webui compatibility)
        msg = ActuatorState()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._base_frame
        msg.estop = self._estop
        # Approximate throttle/brake/steer from measured wheel RPMs so webui
        # UI reflects reality, not command
        avg_mps = ((l_rpm + r_rpm) / 2.0) * self._m_per_rev / 60.0
        diff_mps = ((r_rpm - l_rpm)) * self._m_per_rev / 60.0
        throttle = max(0.0, min(1.0, avg_mps / self._max_v))
        brake = max(0.0, min(1.0, -avg_mps / self._max_v))
        steer = (
            max(-1.0, min(1.0, (diff_mps / self._track_w) / self._max_w))
            if self._max_w > 0 else 0.0
        )
        msg.throttle = throttle
        msg.brake = brake
        msg.steer = steer
        msg.mode = 'D' if not self._estop else 'N'
        msg.watchdog_active = False
        self._state_pub.publish(msg)

        # Integrate + publish wheel odometry
        self._publish_odom(now, l_rpm, r_rpm)

    def _publish_odom(self, now, l_rpm, r_rpm):
        dt = (now - self._odom_last_t).nanoseconds / 1e9
        if dt <= 0 or dt > 0.5:
            self._odom_last_t = now
            return
        self._odom_last_t = now

        l_mps = l_rpm * self._m_per_rev / 60.0
        r_mps = r_rpm * self._m_per_rev / 60.0
        v = (l_mps + r_mps) / 2.0
        w = (r_mps - l_mps) / self._track_w

        # Trapezoidal (midpoint) pose integration: use the average of
        # old and new yaw for the position update. 2nd-order accurate;
        # the previous forward-Euler form over-rotated displacement on turns.
        yaw_delta = w * dt
        yaw_avg = wrap_angle(self._odom_yaw + yaw_delta / 2.0)
        self._odom_yaw = wrap_angle(self._odom_yaw + yaw_delta)
        self._odom_x += v * math.cos(yaw_avg) * dt
        self._odom_y += v * math.sin(yaw_avg) * dt

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = self._odom_x
        odom.pose.pose.position.y = self._odom_y
        odom.pose.pose.orientation.z = math.sin(self._odom_yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self._odom_yaw / 2.0)
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        odom.pose.covariance = self._pose_cov
        odom.twist.covariance = self._twist_cov
        self._odom_pub.publish(odom)

    # ------------------------------------------------------------- serial I/O
    def _serial_write(self, line: str):
        try:
            with self._serial_lock:
                self._serial.write((line + '\n').encode('ascii'))
        except Exception as e:
            self.get_logger().error(f'serial write failed: {e}')

    def _serial_reader(self):
        """
        Read encoder lines and Teensy acknowledgements in the background.

        E lines carry encoder feedback (50 Hz from firmware).
        OK K... lines are the Teensy's acknowledgement of a PID gain write —
        after a `KP0.0008`-style command, the Teensy responds with
        `OK KP=0.00080000` to confirm it pushed the value to BOTH SparkMAXes
        via CAN PARAMETER_WRITE (see firmware/teensy_diff_drive.ino:283).
        We log those at INFO so a tuner can see end-to-end confirmation.
        """
        import re
        E_RE = re.compile(r"E L(-?\d+) (-?[\d.]+) R(-?\d+) (-?[\d.]+)")
        OK_RE = re.compile(r"OK (K[PIDF Z]|A[01]|S|UL=|BURN|L=).*")
        ERR_RE = re.compile(r"ERR .*")
        buf = ''
        while self._running:
            try:
                chunk = self._serial.read(256).decode('ascii', errors='replace')
                if not chunk:
                    continue
                buf += chunk
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    s = line.strip()
                    m = E_RE.match(s)
                    if m:
                        with self._fb_lock:
                            self._l_meas_rpm = float(m.group(1))
                            self._l_meas_pos = float(m.group(2))
                            self._r_meas_rpm = float(m.group(3))
                            self._r_meas_pos = float(m.group(4))
                        continue
                    if OK_RE.match(s):
                        self.get_logger().info(f'  <- Teensy ack: {s}')
                        continue
                    if ERR_RE.match(s):
                        self.get_logger().warn(f'  <- Teensy ERR: {s}')
                        continue
            except Exception as e:
                self.get_logger().error(f'serial reader: {e}')
                time.sleep(0.1)

    # ------------------------------------------------------------- shutdown
    def destroy_node(self):
        self.get_logger().info('shutdown — sending S')
        self._running = False
        try:
            self._serial_write('S')
            time.sleep(0.1)
            self._serial.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
