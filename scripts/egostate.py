#!/usr/bin/env python3.11
"""Print: <speed> <x> <y>  for the ego (vehicle.lincoln.mkz). Standalone
(no nested-heredoc fragility)."""
import carla
c = carla.Client('localhost', 2000)
c.set_timeout(20.0)
w = c.get_world()
egos = [a for a in w.get_actors().filter('vehicle.lincoln.mkz')]
if not egos:
    print("0 0 0")
else:
    e = egos[0]
    v = e.get_velocity()
    l = e.get_transform().location
    spd = (v.x * v.x + v.y * v.y) ** 0.5
    print(f"{spd:.2f} {l.x:.2f} {l.y:.2f}")
