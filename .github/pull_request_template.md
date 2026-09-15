## What changed

<!-- One or two sentences. Link the ISSUES.md entry if this fixes one. -->

## Why

<!-- The problem this solves, or the behavior that was wrong. -->

## How I tested it

<!-- Tick everything you actually ran. CI only covers the first box. -->

- [ ] Offline tests pass (`PYTHONPATH=. python3 -m pytest test -q`)
- [ ] Ran it in CARLA
- [ ] Ran it on the car (dinosaur)
- [ ] Not tested beyond the above — explain below

<!-- Paste relevant output, or point at a file in logs/results/. -->

## Needs attention

<!-- Delete what doesn't apply. -->

- Touches a config that must match Nav2 (`offroad_cost`, `obstacle_threshold`, `raytrace_range_m`, `obstacle_max_range`) — see CONTRIBUTING.md
- Needs hardware or a person to verify before merge
- Depends on another PR: #
