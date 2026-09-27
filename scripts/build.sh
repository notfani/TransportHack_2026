#!/usr/bin/env bash
set -eo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace"
source /opt/ros/humble/setup.bash
colcon build --base-paths src --packages-up-to tram_odometry "$@"
