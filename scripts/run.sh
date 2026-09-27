#!/usr/bin/env bash
set -eo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace"
source /opt/ros/humble/setup.bash
source "$workspace/install/setup.bash"
prefix="$(ros2 pkg prefix tram_odometry)"
if [[ "$prefix" != "$workspace/install/tram_odometry" ]]; then
  echo "Unexpected tram_odometry package: $prefix" >&2
  exit 1
fi
exec ros2 launch tram_odometry run.launch.py "$@"
