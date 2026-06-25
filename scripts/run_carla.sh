#!/bin/bash
# Launch CARLA UE5 via the prebuilt editor with native ROS2.
# First launch compiles shaders (slow).
export CARLA_UNREAL_ENGINE_PATH=/home/merabro/UnrealEngine5_carla
EDITOR=/home/merabro/UnrealEngine5_carla/Engine/Binaries/Linux/UnrealEditor
PROJECT=/home/merabro/CarlaUE5/Unreal/CarlaUnreal/CarlaUnreal.uproject
exec "$EDITOR" "$PROJECT" -vulkan -RenderOffScreen --ros2
