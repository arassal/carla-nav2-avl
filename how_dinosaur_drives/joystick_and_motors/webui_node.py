"""AVROS Web UI node: phone joystick -> ActuatorCommand.

Ported from ~/AV2.1-API/webui/server_standalone.py.
Replaces raw UDP with ROS2 ActuatorCommand publishing.

FastAPI + WebSocket server (uvicorn) in main thread,
rclpy.spin in a background daemon thread.
"""

import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
from avros_msgs.msg import ActuatorCommand, ActuatorState
from std_msgs.msg import Bool


# Latched QoS for /autonomous_mode — matches actuator_node + mission_manager so
# the §I.2 safety-light state survives a subscriber (re)start.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn


class WebUINode(Node):
    """ROS2 node that bridges WebSocket joystick input to ActuatorCommand."""

    def __init__(self):
        super().__init__('webui_node')

        self.declare_parameter('web_port', 8000)
        self.declare_parameter('ssl_certfile', '')
        self.declare_parameter('ssl_keyfile', '')
        self.declare_parameter('max_throttle', 0.55)

        self.web_port = self.get_parameter('web_port').value
        self.ssl_certfile = self.get_parameter('ssl_certfile').value
        self.ssl_keyfile = self.get_parameter('ssl_keyfile').value
        self.max_throttle = self.get_parameter('max_throttle').value

        self._cmd_pub = self.create_publisher(
            ActuatorCommand, '/avros/actuator_command', 10
        )
        # IGVC §I.2 safety light: operator AUTONOMOUS toggle -> latched flag that
        # actuator_node forwards to the Teensy (A1/A0). Lets a judge watch the
        # solid->flash->solid transition during qualification.
        self._auto_pub = self.create_publisher(
            Bool, '/autonomous_mode', LATCHED_QOS
        )
        self._state_sub = self.create_subscription(
            ActuatorState, '/avros/actuator_state',
            self._state_callback, 10
        )

        self._state_lock = threading.Lock()
        self._latest_state = {
            'e': 0, 't': 0.0, 'b': 0.0, 's': 0.0, 'm': 'N', 'w': 0
        }

        self.get_logger().info(
            f'WebUI node started (port={self.web_port})'
        )

    def _state_callback(self, msg: ActuatorState):
        with self._state_lock:
            self._latest_state = {
                'e': 1 if msg.estop else 0,
                't': msg.throttle,
                'b': msg.brake,
                's': msg.steer,
                'm': msg.mode,
                'w': 1 if msg.watchdog_active else 0,
            }

    def get_state(self) -> dict:
        with self._state_lock:
            return dict(self._latest_state)

    def publish_command(
        self, throttle: float, brake: float, steer: float,
        mode: str, estop: bool
    ):
        msg = ActuatorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.estop = estop
        msg.throttle = min(max(0.0, throttle), self.max_throttle)
        msg.brake = min(max(0.0, brake), 1.0)
        msg.steer = min(max(-1.0, steer), 1.0)
        msg.mode = mode
        self._cmd_pub.publish(msg)

    def publish_estop(self):
        msg = ActuatorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.estop = True
        msg.throttle = 0.0
        msg.brake = 1.0
        msg.steer = 0.0
        msg.mode = 'N'
        self._cmd_pub.publish(msg)

    def publish_autonomous(self, value: bool):
        self._auto_pub.publish(Bool(data=bool(value)))


def create_app(node: WebUINode) -> FastAPI:
    """Build the FastAPI application wired to the ROS2 node."""
    app = FastAPI()
    static_dir = os.path.join(
        get_package_share_directory('avros_webui'), 'static'
    )

    active_controller = None
    controller_lock = threading.Lock()

    @app.get('/')
    async def root():
        return FileResponse(os.path.join(static_dir, 'index.html'))

    @app.websocket('/ws')
    async def websocket_endpoint(websocket: WebSocket):
        nonlocal active_controller

        # Check-and-claim under the lock WITHOUT awaiting: controller_lock is
        # a threading.Lock, so blocking on it inside a coroutine stalls the
        # whole event loop. Holding it across an await deadlocks the server
        # (observed 2026-07-02: two devices connecting concurrently wedged
        # uvicorn's accept loop; every later connection hung "Connecting...").
        with controller_lock:
            rejected = active_controller is not None
            if not rejected:
                active_controller = websocket

        if rejected:
            await websocket.accept()
            await websocket.send_json(
                {'error': 'Another controller is connected'}
            )
            await websocket.close(code=1008)
            return

        try:
            await websocket.accept()
        except Exception:
            # never leak the claim if the handshake dies mid-flight
            with controller_lock:
                active_controller = None
            raise

        estop = False
        mode = 'N'
        autonomous = False

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get('type')

                if msg_type == 'control':
                    # nipplejs vector: screen coords (x right = +, y up = +).
                    # ActuatorCommand uses REP-103 (steer + = CCW / left).
                    x = float(data.get('x', 0))
                    y = float(data.get('y', 0))
                    throttle = max(0.0, y)
                    brake = max(0.0, -y)
                    steer = -x
                    node.publish_command(throttle, brake, steer, mode, estop)

                elif msg_type == 'estop':
                    estop = data.get('value', False)
                    if estop:
                        node.publish_estop()
                        # E-stop is not autonomous — force the §I.2 light solid.
                        if autonomous:
                            autonomous = False
                            node.publish_autonomous(False)
                    else:
                        node.publish_command(0.0, 0.0, 0.0, mode, False)

                elif msg_type == 'mode':
                    new_mode = data.get('value', 'N')
                    if new_mode in ('N', 'D', 'S', 'R'):
                        mode = new_mode

                elif msg_type == 'autonomous':
                    # §I.2 safety-light toggle. When ON, the client stops
                    # streaming 'control' so /avros/actuator_command goes stale
                    # and Nav2's /cmd_vel takes over (if running).
                    autonomous = bool(data.get('value', False))
                    node.publish_autonomous(autonomous)
                    node.get_logger().info(
                        f'webui autonomous toggle -> {autonomous}'
                    )

                # 'keepalive' and any other type fall through: no command is
                # published (so actuator_command can go stale under autonomy),
                # we just return current state below.
                await websocket.send_json(node.get_state())

        except WebSocketDisconnect:
            pass
        finally:
            with controller_lock:
                active_controller = None
            node.publish_estop()
            # Controller gone -> not autonomous -> light solid.
            node.publish_autonomous(False)
            node.get_logger().warn('WebSocket disconnected — e-stop + light solid')

    @app.middleware('http')
    async def no_cache(request, call_next):
        response = await call_next(request)
        if request.url.path.endswith(('.js', '.css')):
            response.headers['Cache-Control'] = 'no-store'
        return response

    app.mount(
        '/static',
        StaticFiles(directory=static_dir, follow_symlink=True),
        name='static',
    )

    return app


def main(args=None):
    rclpy.init(args=args)
    node = WebUINode()

    # Spin ROS2 in a background daemon thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = create_app(node)

    ssl_kwargs = {}
    if node.ssl_certfile and node.ssl_keyfile:
        ssl_kwargs['ssl_certfile'] = node.ssl_certfile
        ssl_kwargs['ssl_keyfile'] = node.ssl_keyfile

    try:
        uvicorn.run(
            app, host='0.0.0.0', port=node.web_port,
            log_level='info', **ssl_kwargs
        )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
