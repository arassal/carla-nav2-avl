#!/usr/bin/env python3
"""
CARLA Bridge - Original implementation
Manages CARLA simulation, ego vehicle, and ROS2 interface
"""

import carla
import numpy as np
import struct
import socket
import time
import yaml
from pathlib import Path


class CARLABridge:
    """Bridge between CARLA simulator and ROS2 system"""

    def __init__(self, config_path="sensors.yaml"):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()

        self.ego = None
        self.sensors = {}
        self.config = self._load_config(config_path)

        print(f"✅ Connected to {self.world.get_map().name}")

    def _load_config(self, config_path):
        """Load sensor configuration from YAML"""
        config_file = Path(__file__).parent / config_path
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        return {"town": "Town10HD_Opt", "spawn_point_index": 0}

    def spawn_ego(self):
        """Spawn ego vehicle"""
        # Clean old actors
        for actor in self.world.get_actors():
            if any(actor.type_id.startswith(p) for p in ("vehicle.", "walker.", "sensor.")):
                try:
                    actor.destroy()
                except:
                    pass

        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.filter("vehicle.lincoln.mkz_2020")[0]

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_idx = self.config.get("spawn_point_index", 0)
        spawn_point = spawn_points[spawn_idx % len(spawn_points)]

        self.ego = self.world.spawn_actor(ego_bp, spawn_point)
        print(f"✅ Ego vehicle spawned at index {spawn_idx}")

    def spawn_sensors(self):
        """Spawn configured sensors"""
        sensors_config = self.config.get("sensors", [])

        for sensor_config in sensors_config:
            sensor_type = sensor_config["type"]
            ros_name = sensor_config["ros_name"]
            xyz = sensor_config["xyz"]

            bp_lib = self.world.get_blueprint_library()
            bp = bp_lib.find(sensor_type)

            # Apply attributes
            for attr_name, attr_value in sensor_config.get("attrs", {}).items():
                bp.set_attribute(attr_name, attr_value)

            # Attach to ego
            transform = carla.Transform(carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))
            sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego)

            self.sensors[ros_name] = sensor
            print(f"✅ {ros_name} sensor ready")

    def tick(self):
        """Single world tick"""
        self.world.tick()

    def run(self, duration=None):
        """Main loop"""
        self.spawn_ego()
        self.spawn_sensors()

        print("\n🚀 CARLA bridge running...")
        start_time = time.time()
        tick_count = 0

        try:
            while True:
                self.tick()
                tick_count += 1

                if tick_count % 100 == 0:
                    print(f"  Ticks: {tick_count} | Ego pos: {self.ego.get_location()}")

                if duration and (time.time() - start_time) > duration:
                    break

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n✅ Bridge stopped")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up actors"""
        for sensor in self.sensors.values():
            try:
                sensor.destroy()
            except:
                pass

        if self.ego:
            try:
                self.ego.destroy()
            except:
                pass


if __name__ == "__main__":
    bridge = CARLABridge()
    bridge.run()
