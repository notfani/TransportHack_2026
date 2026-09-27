"""Integration tests of the installed Python node on isolated ROS topics."""
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from ament_index_python.packages import get_package_prefix
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

BASE = 1_700_000_000_000_000_000


def ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


@pytest.fixture(scope="module")
def context():
    ctx = Context()
    rclpy.init(args=[], context=ctx)
    yield ctx
    ctx.shutdown()


class Rig:
    def __init__(self, context, folder, params):
        self.node = Node("integration_" + uuid.uuid4().hex, context=context)
        self.executor = SingleThreadedExecutor(context=context)
        self.executor.add_node(self.node)
        self.prefix = "/integration_" + uuid.uuid4().hex
        self.process = None
        self.log = None
        self.log_path = folder / "node.log"
        self.stamp = BASE
        self.velocity, self.position, self.diagnostics = [], [], []
        topics = {
            "/vehicle/front_bogie_velocity": "front",
            "/vehicle/rear_bogie_velocity": "rear",
            "/vehicle/driver_position_cmd": "command",
            "/sensing/gnss/master/fix": "fix",
            "/result/velocity": "velocity",
            "/result/position": "position",
            "/result/diagnostics": "diagnostics",
            "/clock": "clock",
        }
        self.pubs = {
            name: self.node.create_publisher(kind, self.prefix + "/" + name, 10)
            for name, kind in (("front", VelocitySensor), ("rear", VelocitySensor),
                               ("command", DriverControllerCommand), ("fix", NavSatFix),
                               ("clock", Clock))
        }
        for name, kind, destination in (("velocity", VelocitySensor, self.velocity),
                                        ("position", Odometry, self.position),
                                        ("diagnostics", DiagnosticArray, self.diagnostics)):
            self.node.create_subscription(kind, self.prefix + "/" + name, destination.append, 200)
        options = {"use_sim_time": True, "require_sim_time": True, "start_mode": "relative"}
        options.update(params)
        executable = Path(get_package_prefix("tram_odometry")) / "lib/tram_odometry/odometry_node"
        args = [str(executable), "--ros-args", "--log-level", "error",
                "-r", "__node:=team_test_" + uuid.uuid4().hex]
        for old, new in topics.items():
            args += ["-r", old + ":=" + self.prefix + "/" + new]
        for key, value in options.items():
            args += ["-p", key + ":=" + (str(value).lower() if isinstance(value, bool) else str(value))]
        try:
            self.log = self.log_path.open("w")
            self.process = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, ROS_LOG_DIR=str(folder / "ros_logs")))
            self.until(lambda: all(self.pubs[k].get_subscription_count() > 0
                                   for k in ("front", "rear", "command", "clock"))
                       and self.node.count_publishers(self.prefix + "/velocity") > 0
                       and self.node.count_publishers(self.prefix + "/position") > 0, 8)
            self.clock()
            self.spin(.12)
        except BaseException:
            self.close()
            raise

    def spin(self, duration=.04):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            assert self.process.poll() is None, self.log_path.read_text()
            self.executor.spin_once(timeout_sec=min(.01, max(0, end-time.monotonic())))

    def until(self, condition, timeout=2):
        end = time.monotonic() + timeout
        while not condition() and time.monotonic() < end:
            self.spin(.02)
        assert condition(), "ROS readiness/output timeout: " + self.log_path.read_text()

    def clock(self):
        msg = Clock()
        msg.clock.sec, msg.clock.nanosec = divmod(self.stamp + 3_000_000_000, 1_000_000_000)
        self.pubs["clock"].publish(msg)

    def send(self, name, value, stamp=None):
        msg = DriverControllerCommand() if name == "command" else VelocitySensor()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(
            self.stamp if stamp is None else stamp, 1_000_000_000)
        if name == "command":
            msg.position = int(value)
        else:
            msg.velocity = float(value)
        self.pubs[name].publish(msg)

    def tick(self, dt=.05, **values):
        self.stamp += int(dt * 1e9)
        self.clock()
        self.spin(.01)
        for name in ("front", "rear"):
            if name in values:
                self.send(name, values[name])
        self.spin(.015)
        if "command" in values:
            self.send("command", values["command"])
        self.spin(.025)

    def start(self, speed=18.0):
        # DDS discovery does not guarantee the first /clock sample was delivered.
        deadline = time.monotonic() + 5
        while not (self.velocity and self.position) and time.monotonic() < deadline:
            self.clock()
            self.spin(.03)
            self.send("front", speed)
            self.send("rear", speed)
            self.spin(.03)
            self.send("command", 0)
            self.spin(.12)
        assert self.velocity and self.position, "ROS startup timeout: " + self.log_path.read_text()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.log is not None:
            self.log.close()
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()


@pytest.fixture
def rig(context, tmp_path):
    live = []
    def launch(**params):
        folder = tmp_path / str(len(live))
        folder.mkdir()
        value = Rig(context, folder, params)
        live.append(value)
        return value
    yield launch
    for value in reversed(live):
        value.close()


def test_no_inputs_no_estimate(rig):
    r = rig()
    for _ in range(4):
        r.tick()
    assert not r.velocity and not r.position


@pytest.mark.parametrize("unit,raw", [("kmh", 18.0), ("mps", 5.0)])
def test_units_frames_exact_stamps_and_covariance(rig, unit, raw):
    r = rig(wheel_input_unit=unit)
    r.start(raw)
    v, p = r.velocity[0], r.position[0]
    assert v.velocity == pytest.approx(5.0, abs=1e-6)
    assert ns(v) == ns(p) == BASE
    assert v.header.frame_id == "base_link"
    assert p.header.frame_id == "odom_relative"
    assert p.child_frame_id == "base_link"
    assert p.twist.twist.linear.x == v.velocity
    assert all(math.isfinite(x) for x in p.pose.covariance + p.twist.covariance)
    assert p.twist.covariance[7] > 0


def test_watchdog_keeps_output_when_all_inputs_disappear(rig):
    r = rig()
    r.start()
    for _ in range(24):
        r.tick()
    assert len(r.velocity) >= 12
    stamps = [ns(v) for v in r.velocity]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    assert max(b-a for a, b in zip(stamps, stamps[1:])) <= 100_000_000
    assert all(math.isfinite(v.velocity) for v in r.velocity)
    r.until(lambda: bool(r.diagnostics))
    last = {kv.key: kv.value for kv in r.diagnostics[-1].status[0].values}
    assert last["command_stale"] == "True"


def test_paused_clock_does_not_create_future_states(rig):
    r = rig()
    r.start()
    for _ in range(4):
        r.tick()
    r.spin(.08)
    counts = len(r.velocity), len(r.position)
    r.spin(.25)
    assert (len(r.velocity), len(r.position)) == counts


def test_backward_clock_resets_for_new_replay(rig):
    r = rig()
    r.start()
    for _ in range(4):
        r.tick(front=18, rear=18, command=0)
    r.stamp = BASE - 5_000_000_000
    r.clock()
    r.spin(.1)
    count = len(r.velocity)
    r.send("front", 0)
    r.send("rear", 0)
    r.spin(.03)
    r.send("command", 0)
    r.until(lambda: len(r.velocity) > count)
    assert ns(r.velocity[-1]) == r.stamp
    assert r.velocity[-1].velocity == 0.0
    r.until(lambda: bool(r.position) and ns(r.position[-1]) == r.stamp)
    assert r.position[-1].pose.pose.position.x == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 1e9])
def test_invalid_wheel_with_valid_second_sensor_is_finite(rig, bad):
    r = rig()
    r.send("front", bad)
    r.send("rear", 18.0)
    r.spin(.03)
    r.send("command", 0)
    r.until(lambda: bool(r.velocity))
    assert r.velocity[-1].velocity == pytest.approx(5.0, abs=1e-6)


def test_no_initial_gnss_keeps_late_fix_subscription(rig):
    r = rig(start_mode="gnss", initialization_window_s=.2)
    r.until(lambda: r.pubs["fix"].get_subscription_count() == 1)
    r.start(0)
    for _ in range(8):
        r.tick(front=0, rear=0, command=0)
    assert r.pubs["fix"].get_subscription_count() == 1
    assert r.position[-1].header.frame_id == "odom_relative"


def test_explicit_route_initialization_uses_map_and_no_gnss(rig):
    r = rig(start_mode="route_chainage", route_name="tal_shu", start_chainage_m=10.0)
    r.start(0)
    p = r.position[-1]
    assert p.header.frame_id == "pathgraph"
    assert abs(p.pose.pose.position.x) > 1000
    assert p.pose.covariance[0] >= 9
    assert r.pubs["fix"].get_subscription_count() == 0


def test_timing_diagnostics_cover_command_and_watchdog_outputs(rig):
    r = rig()
    r.start()
    for _ in range(22):
        r.tick(front=18, rear=18, command=0)
    for _ in range(24):
        r.tick()
    r.until(lambda: bool(r.diagnostics))
    d = {kv.key: kv.value for kv in r.diagnostics[-1].status[0].values}
    total = 0
    for kind in ("command", "watchdog"):
        prefix = kind + "_callback_to_publish_"
        count = int(d[prefix+"count"])
        assert count > 0
        assert 0 <= float(d[prefix+"mean_ms"]) <= float(d[prefix+"max_ms"])
        assert 0 <= int(d[prefix+"over_100ms_count"]) <= count
        total += count
    assert total <= len(r.velocity)
    assert d["callback_work_is_end_to_end_latency"] == "False"
