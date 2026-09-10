# logs/ — terminal output, benchmark data, run captures

Scratch space for evidence. When something is broken (or fixed), the terminal
output that proves it goes here so a claim in `ISSUES.md` or a PR description
can point at a file instead of a memory.

## Layout

| dir | what goes in it | committed? |
|---|---|---|
| `tests/`   | pytest / colcon output, full runs not just the tail | no |
| `bench/`   | `bench_perception.py`, `ros2 topic hz`, timing tables | no |
| `runs/`    | live session captures: node stdout, `ros2 bag` notes, on-car logs | no |
| `results/` | the curated numbers worth keeping — a short `.md` per finding | **yes** |

Raw output under `tests/`, `bench/`, and `runs/` is gitignored: it is large,
machine-specific, and regenerable. Anything that should survive into a PR gets
summarized into `results/` as markdown and committed.

## Naming

    <YYYY-MM-DD>_<what>_<host>.log

e.g. `2026-09-10_pytest-perception-costmap_chris-linux.log`. The date and host
matter — a Jetson number and a laptop number are not comparable, and half the
open questions in `DEPLOY.md` are about exactly that difference.

## Capturing

Keep both the terminal view and the file:

    <command> 2>&1 | tee logs/tests/2026-09-10_pytest_chris-linux.log

Record the environment alongside the output when it could matter (ROS distro,
whether ROS was sourced, GPU present, numpy version) — several issues in
`ISSUES.md` only reproduce with or without `source /opt/ros/*/setup.bash`.
