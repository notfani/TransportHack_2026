# Резервная одометрия трамвая

Пакеты ROS 2 Humble: `tram_vehicle_msgs`, `tram_odometry`.
Оценка продольной скорости и положения по ручке контроллера и скоростям двух тележек. GNSS только для начальной выставки.

Требования: Ubuntu 22.04, ROS 2 Humble. Сборка без сети и без pip-пакетов (нет NumPy/CatBoost/PyTorch в рантайме).

## Сборка

```bash
cd TransportHack_2026   # корень этого репозитория
source /opt/ros/humble/setup.bash
bash scripts/build.sh
bash scripts/test.sh
```

## Прогон bag

Подставьте путь к каталогу rosbag2 организаторов (`metadata.yaml` + `*.db3`):

```bash
bash scripts/run.sh bag:=/путь/к/bag
```

В другом терминале:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /result/velocity
ros2 topic echo /result/position
ros2 topic hz /result/velocity
```

Контрольный короткий прогон: `30618_bab2fe58`. Длинный: `30618_2050d396`.

Документы сдачи: `docs/INSTRUKCIYA.md`, `MODEL.md`, `ASSUMPTIONS.md`, `ACCURACY.md`, `LIMITATIONS.md`.
Тексты полей формы: `docs/FORM.txt`.
