#!/usr/bin/env bash
set -eo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace"
printf '[test.sh] Проверяю ROS 2 и установленную сборку...\n'
source /opt/ros/humble/setup.bash
if [[ ! -f install/setup.bash ]]; then
  printf 'Сначала выполните bash scripts/build.sh\n' >&2
  exit 2
fi
source install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-92}"
printf '[test.sh] Запускаю текущие Python и ROS тесты (ROS_DOMAIN_ID=%s).\n' "$ROS_DOMAIN_ID"
python3 -m pytest -v -s src/tram_odometry/tests "$@"
printf '[test.sh] Все тесты прошли.\n'
