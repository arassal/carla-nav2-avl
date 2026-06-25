#!/bin/bash
# Run the built CARLA UE5 game binary as a server with native ROS2.
export CARLA_UNREAL_ENGINE_PATH=/home/merabro/UnrealEngine5_carla
export LD_LIBRARY_PATH=/home/merabro/UnrealEngine5_carla/Engine/Binaries/Linux:/home/merabro/CarlaUE5/Unreal/CarlaUnreal/Binaries/Linux:${LD_LIBRARY_PATH}
GAME=/home/merabro/CarlaUE5/Unreal/CarlaUnreal/Binaries/Linux/CarlaUnreal
PROJECT=/home/merabro/CarlaUE5/Unreal/CarlaUnreal/CarlaUnreal.uproject
exec "$GAME" "$PROJECT" /Game/Carla/Maps/Town10HD_Opt \
  -vulkan -nosound --ros2 -carla-rpc-port=2000
