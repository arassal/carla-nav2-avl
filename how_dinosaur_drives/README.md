# How Dinosaur Drives Itself (Plain-English Guide)

This folder pulls together **everything on the robot that's involved in making it move** — whether a person is driving it with a phone, or it's driving itself. It's meant to be readable by someone who has never touched this codebase before.

The robot's brain is a small onboard computer (an NVIDIA Jetson, nicknamed "dinosaur"). Everything below runs on that computer.

There are 4 pieces. Think of them like a chain — information flows from one to the next:

```
 1. SEE               2. LOCATE              3. DECIDE                4. MOVE
 (cameras + lidar)  →  (where am I?)      →  (where should I go?)  →  (turn the wheels)
 perception/          lidar/                 path_planning/           joystick_and_motors/
 (already in this
  repo, one level up
  at ros2_ws/src/
  perception_costmap)
```

You don't have to read these in order, but if you're new, reading them in this order (1 → 4) will make the most sense.

---

## 1. Seeing — `perception` (cameras + AI)

**Where:** this is already elsewhere in this same repository, at `ros2_ws/src/perception_costmap/`. It's not duplicated into this folder.

**What it does, in plain words:** the robot has 3 cameras (front, left, right) and looks at the video with an AI model (like the AI in a self-driving car) to spot obstacles — people, cones, other vehicles — and figure out which parts of the ground are safe to drive on. It turns all of that into a simple grid map, like a piece of graph paper laid on the ground around the robot, where each square is marked "safe," "dangerous," or "not sure yet."

## 2. Locating — `lidar/`

**What lidar is:** a spinning sensor on top of the robot that shoots out laser beams and measures how long they take to bounce back. This tells the robot exactly how far away everything around it is, in every direction, all the time — like echolocation, but with light instead of sound.

**Files in this folder:**
- `sensors.launch.py` — the "on switch" script that turns on the lidar (a Velodyne VLP-16) and the robot's inertial/GPS sensor (an Xsens unit that tells the robot which way it's facing and roughly where it is on Earth) at the same time.
- `velodyne.yaml` — the settings file for the lidar (spin speed, which laser channels to trust, etc.)

**Why it matters:** the cameras are good at recognizing *what* something is, but lidar is much more precise about *exactly how far away* it is. The robot uses both together.

## 3. Deciding where to go — `path_planning/`

This is the "GPS navigation app" part of the robot — it looks at the safe/unsafe grid map from step 1, the robot's current position from step 2, and figures out a path to wherever it's supposed to go, the same way a phone maps app draws a route, except it's re-drawing that route many times per second and steering around obstacles as they appear.

**Files in this folder, plain-English version:**
- `auto_drive.launch.py` — turns on "drive yourself" mode. You click a destination point on a map on the operator's screen, and the robot drives itself there.
- `navigation.launch.py` — turns on the full self-driving system (this is the industry-standard self-driving toolkit for robots, called "Nav2").
- `nav2_params_auto_drive.yaml` — the settings for how the self-driving system behaves: how fast it's allowed to go, how cautious it is around obstacles, how often it re-plans its route.
- `cpp_campus_graph.geojson` — a hand-made map of the Cal Poly Pomona campus, stored as a set of connected dots (like stops on a subway map) representing the paths the robot is allowed to drive on.
- `campus_navigator.py` — reads that campus map and works out step-by-step directions from where the robot is to where it needs to go.
- `route_utils.py` / `graph_validation.py` — helper tools that check the campus map for mistakes (like a subway map with a broken connection) before the robot tries to use it.
- `autonomy_monitor.py` — a safety watcher. If something goes wrong while the robot is driving itself (it loses track of where it is, a sensor stops responding, etc.), this is what notices and reacts.
- `route_graph_viz.py` — draws the campus map and the robot's planned route on screen, so a human operator can see what the robot is "thinking."

## 4. Moving — `joystick_and_motors/`

This is the last link in the chain: turning a decision ("go forward," "turn left") into the wheels actually spinning. It's also what a **human** uses to drive the robot directly from a phone, bypassing steps 1–3 entirely.

**The phone joystick (remote control):**
- `index.html` / `app.js` — this is the actual webpage you open on a phone to drive the robot. It shows an on-screen joystick, an E-STOP button, and a few driving-mode buttons. No app install needed — it's just a webpage.
- `webui_node.py` — the program on the robot that the phone webpage talks to. It listens for "move the stick this way" messages from your phone and turns them into drive commands.
- `webui.launch.py` / `webui_params.yaml` — the "on switch" and settings (which network port to use, the safety certificate, a speed limit cap) for the phone-control system.

**The motor controller (turns commands into actual wheel movement):**
- `actuator_node.py` — the most important file in this folder. It takes a drive command (from either the phone joystick OR the self-driving system in `path_planning/`) and converts it into the exact instructions the wheel motors need — how fast each side should spin to go in the requested direction. It sends those instructions over a wire to a small onboard controller board, which relays them to the actual motors.
- `actuator_params.yaml` — settings for the motors: top speed, how gently it speeds up/slows down (so nobody gets jolted), physical measurements of the robot needed to calculate turns correctly.
- `command_arbitration.py` — the referee. Since drive commands can come from *either* the phone joystick *or* the self-driving system, this decides who's "in control" at any given moment, so the two never fight over the steering wheel at the same time.

---

## The short version

**A person drives with their phone** → phone webpage → `webui_node.py` → `actuator_node.py` → wheels turn.

**The robot drives itself** → cameras + lidar build a safety map → `path_planning/` decides a route → `actuator_node.py` → wheels turn.

Either way, `actuator_node.py` is the final common step that actually makes the car move — everything above it is just different ways of deciding *what* to tell it.
