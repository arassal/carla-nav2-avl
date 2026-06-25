# AVL CARLA + Nav2 Project - Contribution Guide

## Project Overview

Professional autonomous navigation research platform combining:
- **CARLA 0.10.0** (Unreal Engine 5) simulator
- **ROS2 Humble** middleware
- **Navigation2** (Nav2) stack
- **Lane following** with traffic light enforcement
- **Obstacle detection** and emergency braking

## Repository Structure

```
carla-nav2-avl/
├── main                          # Production-ready code
├── develop                        # Integration branch for features
├── feature/alexander              # alexander's feature branch
├── feature/jchy05                 # jchy05's feature branch
├── feature/adamcastillo07         # AdamCastillo07's feature branch
├── feature/ad-tap                 # adrian's feature branch
├── bugfix/improvements            # Bug fixes and improvements
└── docs/setup-guides              # Documentation and setup guides
```

## Team Members

| GitHub | Branch | Role |
|--------|--------|------|
| @arassal (alexander) | `feature/alexander` | Project lead, CARLA integration |
| @jchy05 | `feature/jchy05` | Nav2 planning and optimization |
| @AdamCastillo07 | `feature/adamcastillo07` | Visualization and monitoring |
| @Ad-Tap (adrian) | `feature/ad-tap` | Control systems and tuning |

## Workflow

### 1. **Start Your Work**

```bash
# Clone the repo
git clone https://github.com/arassal/carla-nav2-avl.git
cd carla-nav2-avl

# Checkout your feature branch
git checkout feature/YOUR_USERNAME

# Pull latest changes
git pull origin feature/YOUR_USERNAME
```

### 2. **Make Changes**

```bash
# Create a feature branch from your branch
git checkout -b feature/YOUR_USERNAME/your-feature

# Make changes and commit
git add .
git commit -m "Add/Fix: Clear description of changes"

# Push to your feature branch
git push origin feature/YOUR_USERNAME/your-feature
```

### 3. **Create Pull Request to Develop**

When your feature is ready:

1. Push to your feature branch
2. Go to: https://github.com/arassal/carla-nav2-avl/pulls
3. Click "New Pull Request"
4. Select:
   - **Base**: `develop`
   - **Compare**: `feature/YOUR_USERNAME/your-feature`
5. Add description and request review

### 4. **Code Review**

- Describe what you changed
- Link relevant issues or tests
- Request review from other team members
- Address feedback and update PR

### 5. **Merge to Main**

After PR is merged to `develop`:
- Wait for validation testing
- Final PR from `develop` → `main`
- Only arassal (alexander) merges to main

## Branch Naming Convention

```
feature/USERNAME/what-you-did        # New features
bugfix/what-you-fixed                # Bug fixes
docs/what-you-documented             # Documentation
refactor/what-you-improved           # Code improvements
```

## Commit Message Style

```
[TYPE] Brief description

Optional longer explanation if needed.
- Add details about changes
- Explain why not just what

Type: Add/Fix/Update/Refactor/Docs
```

Example:
```
Add lane-following pure pursuit gains

Increased lookahead distance from 2.0m to 3.5m for smoother curves.
Updated controller frequency from 10Hz to 20Hz for real-time response.
Tested on Town10HD with traffic - achieves stable 5 m/s lane following.
```

## Testing Before Merge

1. **Local testing** (on your machine with CARLA)
   ```bash
   # Start CARLA
   cd ~/carla
   ./CarlaUE4.sh -quality-level=Low
   
   # Run stack from your branch
   cd ~/selfdrive_carla_ue5_ros2
   git checkout feature/YOUR_USERNAME
   ./scripts/run_stack.sh
   ```

2. **Verify no conflicts** with main branch
   ```bash
   git merge-base --is-ancestor feature/YOUR_USERNAME main
   ```

3. **Run minimal tests** if available
   ```bash
   colcon test
   ```

## Key Files & Areas

| Area | Files | Owner |
|------|-------|-------|
| CARLA Bridge | `world_setup/carla_bridge.py` | alexander |
| Nav2 Config | `scripts/nav2.yaml` | jchy05 |
| Visualization | `controller/sdc_rerun_node.py` | AdamCastillo07 |
| Localization | `controller/carla_localization.py` | All |

## Communication

- **Async**: Use GitHub Issues for discussions
- **Sync**: Weekly syncs in comments on PRs
- **Blockers**: Create issue and tag relevant person

## Setup Instructions

See `docs/setup-guides` branch for:
- Ubuntu 22.04 native install
- Docker setup
- Conda environment
- Troubleshooting guides

## Project Goals

1. ✅ **Baseline**: Working lane-following with Nav2
2. 🔄 **Current**: Obstacle avoidance + traffic light enforcement
3. 📋 **Next**: Multi-vehicle coordination
4. 🎯 **Future**: Full autonomous mission planning

## Emergency Contacts

- **CARLA issues**: Contact arassal
- **Nav2 issues**: Contact jchy05
- **Visualization issues**: Contact AdamCastillo07

## Resources

- [CARLA Documentation](https://carla.readthedocs.io/)
- [ROS2 Documentation](https://docs.ros.org/)
- [Navigation2 Documentation](https://navigation.ros.org/)
- [Contributors](CONTRIBUTORS.md)

---

**Let's build something awesome together! 🚗✨**
