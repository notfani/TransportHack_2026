#!/usr/bin/env python3
"""Exercise actual rosbag --loop and check two complete replay cycles."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

import rclpy
import yaml
from ament_index_python.packages import get_package_prefix
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tram_vehicle_msgs.msg import VelocitySensor
from validate_bag import stamp, stop


class Monitor(Node):
    def __init__(self, prefix):
        super().__init__("loop_monitor_" + uuid.uuid4().hex)
        self.velocity = [[]]
        self.position = [[]]
        self.create_subscription(VelocitySensor, prefix + "/result/velocity",
                                 lambda m: self.receive(self.velocity, m), 10000)
        self.create_subscription(Odometry, prefix + "/result/position",
                                 lambda m: self.receive(self.position, m), 10000)

    @staticmethod
    def receive(cycles, message):
        if cycles[-1] and stamp(message) < stamp(cycles[-1][-1]) - 500_000_000:
            cycles.append([])
        cycles[-1].append(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.bag = args.bag.resolve()
    meta = yaml.safe_load((args.bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    duration = meta["duration"]["nanoseconds"] / 1e9
    args.report.parent.mkdir(parents=True, exist_ok=True)
    prefix = "/loop_" + uuid.uuid4().hex
    inputs = ["/vehicle/front_bogie_velocity", "/vehicle/rear_bogie_velocity",
              "/vehicle/driver_position_cmd", "/sensing/gnss/master/fix"]
    mapping = {t: prefix + t for t in inputs +
               ["/clock", "/result/velocity", "/result/position", "/result/diagnostics"]}
    rclpy.init(args=[])
    monitor = Monitor(prefix)
    worker = player = None
    logs = []
    began = time.monotonic()
    try:
        node_path = args.report.with_suffix(".node.log")
        logs = [node_path.open("w"), args.report.with_suffix(".player.log").open("w")]
        exe = Path(get_package_prefix("tram_odometry")) / "lib/tram_odometry/odometry_node"
        argv = [str(exe), "--ros-args", "-p", "use_sim_time:=true",
                "-r", "__node:=loop_worker_" + uuid.uuid4().hex]
        for old, new in mapping.items():
            argv += ["-r", old + ":=" + new]
        worker = subprocess.Popen(argv, stdout=logs[0], stderr=subprocess.STDOUT,
                                  env=dict(os.environ, ROS_LOG_DIR=str(args.report.parent / "loop_ros_logs")))
        deadline = time.monotonic() + 10
        while (monitor.count_publishers(prefix + "/result/velocity") < 1 or
               monitor.count_subscribers(prefix + "/vehicle/driver_position_cmd") < 1):
            if worker.poll() is not None:
                raise RuntimeError(node_path.read_text())
            if time.monotonic() > deadline:
                raise RuntimeError("Discovery timeout")
            rclpy.spin_once(monitor, timeout_sec=.02)
        player = subprocess.Popen(
            ["ros2", "bag", "play", str(args.bag), "--loop", "--clock", "100",
             "--delay", "3", "--disable-keyboard-controls", "--topics", *inputs,
             "--remap", *[old + ":=" + new for old, new in mapping.items()]],
            stdout=logs[1], stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 2 * (duration + 3) + 30
        print("Waiting for two full cycles and the beginning of a third...", flush=True)
        while len(monitor.velocity) < 3 or len(monitor.position) < 3:
            if worker.poll() is not None or player.poll() is not None:
                raise RuntimeError("Worker/player stopped before two complete loops")
            if time.monotonic() > deadline:
                raise RuntimeError("Two-loop deadline exceeded")
            rclpy.spin_once(monitor, timeout_sec=.02)
        cycles = []
        for i in range(2):
            v, p = monitor.velocity[i], monitor.position[i]
            cycles.append({
                "velocity_count": len(v), "position_count": len(p),
                "velocity_span_s": (stamp(v[-1]) - stamp(v[0])) / 1e9,
                "strictly_increasing_velocity": all(stamp(b) > stamp(a) for a, b in zip(v, v[1:])),
                "strictly_increasing_position": all(stamp(b) > stamp(a) for a, b in zip(p, p[1:])),
                "all_finite": all(math.isfinite(m.velocity) for m in v) and all(
                    math.isfinite(x) for m in p for x in
                    (m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)),
                "frames": sorted(set(m.header.frame_id for m in p)),
                "first_speed_mps": v[0].velocity,
                "last_speed_mps": v[-1].velocity,
                "first_relative_x_m": next((m.pose.pose.position.x for m in p
                                           if m.header.frame_id == "odom_relative"), None),
                "paired_stamps": len({stamp(m) for m in v} & {stamp(m) for m in p}),
            })
        common_v = [{stamp(m): m.velocity for m in c} for c in monitor.velocity[:2]]
        shared = sorted(common_v[0].keys() & common_v[1].keys())
        differences = [abs(common_v[0][s] - common_v[1][s]) for s in shared]
        reset_count = node_path.read_text().count("state reset.")
        checks = {
            "two_complete_cycles": all(c["velocity_span_s"] >= duration - 5 for c in cycles),
            "both_map_anchors_restored": all("pathgraph" in c["frames"] for c in cycles),
            "finite_and_ordered_per_cycle": all(c["all_finite"] and c["strictly_increasing_velocity"]
                                                and c["strictly_increasing_position"] for c in cycles),
            "node_logged_two_resets": reset_count >= 2,
            "relative_distance_restarted": all(c["first_relative_x_m"] is not None and
                                               abs(c["first_relative_x_m"]) < 1 for c in cycles),
        }
        result = {"bag": str(args.bag), "rate": 1.0, "elapsed_wall_s": time.monotonic() - began,
                  "cycles": cycles, "node_reset_count": reset_count, "checks": checks,
                  "passed": all(checks.values()), "common_speed_stamps": len(shared),
                  "repeat_speed_abs_difference_max_mps": max(differences, default=None),
                  "note": "Actual rosbag --loop; stopped deliberately at third-cycle start. Scheduling can differ."}
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        stop(player)
        stop(worker)
        for log in logs:
            log.close()
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()