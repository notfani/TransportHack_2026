# Инструкция для жюри

## Что проверять

Нода `tram_odometry` на ROS 2 Humble. Входы — три топика записи. Выходы:

| топик | тип | поле |
|---|---|---|
| `/result/velocity` | `tram_vehicle_msgs/msg/VelocitySensor` | `velocity`, м/с |
| `/result/position` | `nav_msgs/msg/Odometry` | `pose.pose.position` (м), `twist.twist.linear.x` (м/с) |
| `/result/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | юз, доверие к колёсам, источник оценки, время callback |

`header.stamp` скорости и положения одинаковый и берётся из времени записи, не из часов машины. `header.frame_id` положения: `pathgraph` — привязка к карте маршрута; `odom_relative` — относительная одометрия (нет выставки или вне карты).

## Сборка (без интернета после установки Humble)

```bash
cd TransportHack_2026          # корень репозитория
source /opt/ros/humble/setup.bash
bash scripts/build.sh          # colcon: tram_vehicle_msgs + tram_odometry
bash scripts/test.sh           # контрактные тесты, bag не нужен
```

Зависимости рантайма: rclpy, стандартные сообщения Humble, пакет `tram_vehicle_msgs` из этого архива. `rosdep install --from-paths src --ignore-src --rosdistro humble -r -y` при необходимости.

## Воспроизведение bag

Нужен издатель `/clock` (`ros2 bag play --clock`). Скрипт делает это сам:

```bash
bash scripts/run.sh bag:=/путь/к/30618_bab2fe58
```

Эквивалент вручную: в одном терминале нода (`play_bag:=false`), во втором `ros2 bag play <bag> --clock 100`.

Смотреть выходы: `ros2 topic echo /result/velocity`, `/result/position`. Частота: `ros2 topic hz /result/velocity` (ожидание ≈ 20 Гц).

Автоотчёт (частота, RMSE к GNSS, CPU/RAM, зазор stamp):

```bash
source install/setup.bash
python3 scripts/validate_bag.py /путь/к/bag --rate 1 --report reports/jury.json
```

В JSON: `speed_error_mps`, `velocity_steady_wall.hz`, `resources`, счётчики `/result/*`. На записях без GNSS скорость относительно эталона не считается.

## Задержка

В диагностике и в логе ноды при остановке: `command_callback_to_publish_*` — монотонное время от входа в callback до обеих публикаций. На контрольных прогонах max < 10 мс, счётчик `over_100ms` = 0. Это не DDS-доставка и не разность stamp с wall-clock (stamp — время bag).

## Если нет выхода

1. Оба терминала: `source install/setup.bash`.
2. `ros2 topic echo /clock --once` — без часов нода в режиме bag молчит.
3. Не запускать вторую ноду на тех же `/result/*`.
4. `ROS_DOMAIN_ID` одинаковый во всех терминалах.
