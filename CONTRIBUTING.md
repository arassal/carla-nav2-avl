# Contributing

How to get a change from your laptop into the repo. New to the project?
Read `README.md` first — it explains what the robot is and what's real.

## 1. Get a branch you can push

- **Collaborators** (you can push to `arassal/carla-nav2-avl`): clone it and
  push branches straight to it.
- **Everyone else:** fork it on GitHub, push to your fork, and open the PR
  from there. `gh repo fork arassal/carla-nav2-avl --remote` does both.

Never push directly to `copy` or `main` — every change goes through a PR.

## 2. Branch from the right base

**PRs currently target `copy`**, the branch where active work lands. `main`
is behind it. Which branch becomes the long-term base is still undecided
(`ISSUES.md` D6) — check with Alexander if in doubt.

    git fetch origin
    git switch -c feature/<your-name>-<topic> origin/copy

One topic per branch. A PR that fixes a bug *and* reorganizes docs is two PRs.

## 3. Run the tests

No ROS or GPU needed — only `numpy`, `opencv-python` and `pytest`:

    cd ros2_ws/src/perception_costmap && PYTHONPATH=. python3 -m pytest test -q
    cd driving_seg && PYTHONPATH=. python3 -m pytest test -q

Use `PYTHONPATH=.` exactly. Appending (`PYTHONPATH=.:$PYTHONPATH`) breaks
when ROS is sourced in your shell: pytest picks up ROS's plugins and fails on
`No module named 'lark'`.

CI runs the same two commands on every PR, so a red check means one of them
failed. **Green tests do not mean the car works** — they cover only the
ROS-free modules. Say in the PR what you ran beyond them (CARLA, the car,
nothing).

## 4. Open the PR

Fill in the template GitHub shows you. At least one approving review before
merge; the reviewer checks that the "how tested" section is true.

## Conventions

**Code**
- Python 3.8-compatible syntax — the Jetson floor. No `match`, no `X | Y`
  type unions, no `list[int]`-style annotations. CI checks the syntax.
- `torch`, `ultralytics`, `carla` and ROS message types are optional and
  imported lazily. The tests must pass with only numpy + opencv installed.
- Core modules (`segmentation`, `obstacles`, `bev`, `occupancy`, `temporal`,
  `carla_convert`, `costmap_cloud`, `util`) stay ROS-free; only nodes import
  `rclpy`. That split is what makes them testable offline.
- Grid math is REP-103 (+x forward, +y left), OccupancyGrid row-major, costs
  -1 unknown / 0 free / 100 lethal. CARLA is left-handed (y right) — use the
  conversions in `carla_convert.py`, don't re-derive them.
- No hardcoded `/home/<user>` paths in nodes, launch files or configs. The
  as-run `deploy/*.sh` scripts are the one documented exception.

**Configs that must agree.** Some perception values are only correct if a
Nav2 value matches. Nothing checks this, and getting it wrong fails silently
(`ISSUES.md` C3, C4). If you touch any of these, check the other side:

| perception side | must agree with |
|---|---|
| `offroad_cost` (`perception_*.yaml`) | ≥ `obstacle_threshold` in `deploy/costmap_to_cloud.py`, or road-keeping turns off |
| `raytrace_range_m` (`costmap_to_cloud.py`, 16 m) | Nav2 `obstacle_max_range` must stay below it, in every Nav2 params file that reads `/perception/costmap_cloud` |

**Commits**
- Subject: `<area>: <what changed>` — e.g. `perception_costmap: fix
  world_to_cell truncation`, `deploy: ...`, `repo: ...`.
- Body says *why*, and names any hardware assumption you rely on.
- Small commits, one idea each. No AI co-author trailers.
- Don't rewrite (rebase/force-push) other people's branches.
