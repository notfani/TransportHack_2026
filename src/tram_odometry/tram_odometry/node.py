"""Reference ROS 2 Humble node. Runtime ROS verification remains required."""
from collections import Counter
from dataclasses import fields
import json
import math
import signal
from pathlib import Path
from time import perf_counter

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from tram_estimator import Config
from tram_estimator.drive import TableDrive
from .bridge import Bridge, BridgeConfig
from .hybrid_drive import HybridDrive
from tram_position import CheckedRouteMap


def stamp_ns(message):
    return int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)


def fill_stamp(header, value):
    header.stamp.sec, header.stamp.nanosec = divmod(int(value), 1_000_000_000)


class OdometryNode(Node):
    def __init__(self):
        super().__init__("tram_odometry")
        share = Path(get_package_share_directory("tram_odometry"))
        self.declare_parameter("require_sim_time", True)
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        if self.get_parameter("require_sim_time").value and not self.get_parameter("use_sim_time").value:
            raise ValueError("Replay requires use_sim_time:=true and ros2 bag play --clock 100")
        self.declare_parameter("drive_table_path", str(share / "config" / "drive_table.json"))
        self.declare_parameter("hybrid_config_path", str(share / "config" / "hybrid.json"))
        self.declare_parameter("drive_model_kind", "hybrid")
        self.declare_parameter("route_bundle_path", str(share / "config" / "route_bundle.json"))
        # Keep the legacy master topic parameter name for launch compatibility.
        self.declare_parameter("initial_fix_topic", "/sensing/gnss/master/fix")
        self.declare_parameter("rover_fix_topic", "/sensing/gnss/rover/fix")
        self.declare_parameter("master_velocity_topic", "/sensing/gnss/master/vel")
        self.declare_parameter("route_bundle_sha256", "")
        for f in fields(BridgeConfig):
            self.declare_parameter(f.name, getattr(BridgeConfig(), f.name))
        for f in fields(Config):
            self.declare_parameter("estimator."+f.name, getattr(Config(), f.name))
        self.bridge_config = BridgeConfig(**{f.name: self.get_parameter(f.name).value for f in fields(BridgeConfig)})
        self.core_config = Config(**{f.name: self.get_parameter("estimator."+f.name).value for f in fields(Config)})
        kind=self.get_parameter("drive_model_kind").value
        if kind=="hybrid":
            self.drive=HybridDrive(self.get_parameter("hybrid_config_path").value)
        elif kind=="table":
            self.drive=TableDrive(self.get_parameter("drive_table_path").value)
        else:
            raise ValueError("drive_model_kind must be hybrid or table")
        self.checked_map = CheckedRouteMap.load(
            self.get_parameter("route_bundle_path").value,
            expected_sha256=self.get_parameter("route_bundle_sha256").value or None)
        self.localizer = self.checked_map.as_indexed_localizer()
        self.bridge = Bridge(self.drive, self.localizer, self.bridge_config, self.core_config)
        self.last_diagnostic_ns = 0
        self.publish_timing = {
            kind: {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "over_100ms_count": 0}
            for kind in ("command", "watchdog")}
        self.publish_gaps = {
            kind: {"last_at": None, "max_ms": 0.0, "max_stamp_ns": 0,
                   "over_100ms_count": 0}
            for kind in ("velocity", "position")}
        self.drive_model_counts = Counter()
        self.reset_requested = False
        self.gnss_subscriptions = {}
        self.input_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                                    reliability=ReliabilityPolicy.BEST_EFFORT,
                                    durability=DurabilityPolicy.VOLATILE)
        self.velocity_pub = self.create_publisher(VelocitySensor, "/result/velocity", 10)
        self.position_pub = self.create_publisher(Odometry, "/result/position", 10)
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, "/result/diagnostics", 10)
        self.subscriptions_owned = [
            self.create_subscription(VelocitySensor, "/vehicle/front_bogie_velocity",
                                     lambda message: self.on_wheel("front", message), self.input_qos),
            self.create_subscription(VelocitySensor, "/vehicle/rear_bogie_velocity",
                                     lambda message: self.on_wheel("rear", message), self.input_qos),
            self.create_subscription(DriverControllerCommand, "/vehicle/driver_position_cmd",
                                     self.on_command, self.input_qos),
        ]
        self._sync_gnss_subscription()
        self.timer = self.create_timer(self.bridge_config.watchdog_period_s, self.on_timer)
        self.jump_handle = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-500_000_000),
                          on_clock_change=True), post_callback=self.on_clock_jump)
        self.get_logger().info("Initialized; covariance floors are engineering assumptions. ROS timing gates still require measurement.")

    def on_clock_jump(self, _jump):
        # Keep clock callback short and avoid rebuilding while the clock is locked.
        self.reset_requested = True

    def _prepare_callback(self):
        if self.reset_requested:
            if hasattr(self.drive,"reset_unavailable"):
                self.drive.reset_unavailable()
            self.bridge = Bridge(self.drive, self.localizer, self.bridge_config, self.core_config)
            self.last_diagnostic_ns = 0
            self.reset_requested = False
            self._sync_gnss_subscription()
            self.get_logger().warning("Replay clock changed backwards or clock source changed; state reset.")
        return self.get_clock().now().nanoseconds

    def _sync_gnss_subscription(self):
        if self.bridge.gnss_closed:
            for subscription in self.gnss_subscriptions.values():
                self.destroy_subscription(subscription)
            self.gnss_subscriptions.clear()
        elif not self.gnss_subscriptions:
            # Retain bounded causal inputs for a late first anchor. Once the
            # accepted initial window ends, the initializer rejects corrections.
            self.gnss_subscriptions = {
                "master_fix": self.create_subscription(
                    NavSatFix, self.get_parameter("initial_fix_topic").value,
                    lambda message: self.on_fix(message, "master"), self.input_qos),
                "rover_fix": self.create_subscription(
                    NavSatFix, self.get_parameter("rover_fix_topic").value,
                    lambda message: self.on_fix(message, "rover"), self.input_qos),
                "master_velocity": self.create_subscription(
                    TwistStamped, self.get_parameter("master_velocity_topic").value,
                    self.on_gnss_velocity, self.input_qos),
            }

    def on_wheel(self, name, message):
        clock_ns = self._prepare_callback()
        if clock_ns > 0:
            self.bridge.receive_wheel(name, stamp_ns(message), float(message.velocity), clock_ns)

    def on_command(self, message):
        began = perf_counter()
        clock_ns = self._prepare_callback()
        if clock_ns > 0:
            output = self.bridge.receive_command(stamp_ns(message), int(message.position), clock_ns)
            self.publish(output, began)

    def on_fix(self, message, source="master"):
        clock_ns = self._prepare_callback()
        if clock_ns <= 0:
            return
        # UNKNOWN covariance uses the positive runtime floor. A known large
        # covariance cannot masquerade as a precise initial refinement sample.
        covariance = [float(v) for v in message.position_covariance]
        variance = (max(abs(v) for v in covariance)
            if all(math.isfinite(v) for v in covariance)
            and all(covariance[i] >= 0 for i in (0, 4, 8)) else math.nan)
        self.bridge.receive_fix(stamp_ns(message), message.latitude, message.longitude,
            message.altitude, message.status.status, source=source,
            covariance_type=int(message.position_covariance_type), variance_m2=variance,
            clock_ns=clock_ns)

    def on_gnss_velocity(self, message):
        clock_ns = self._prepare_callback()
        if clock_ns <= 0:
            return
        velocity = message.twist.linear
        # Input vector is ENU, as in the supplied receiver recording contract.
        # It checks initial body-heading consistency, never corrects speed.
        self.bridge.receive_velocity(stamp_ns(message), velocity.x, velocity.y,
            velocity.z, source="master", clock_ns=clock_ns)

    def on_timer(self):
        began = perf_counter()
        output = self.bridge.tick(self._prepare_callback())
        self.publish(output, began)

    def publish(self, output, began):
        if output is None:
            return
        estimate = output.estimate
        velocity = VelocitySensor()
        fill_stamp(velocity.header, estimate.stamp_ns)
        velocity.header.frame_id = self.bridge_config.child_frame_id
        velocity.velocity = float(estimate.speed_mps)
        odometry = Odometry()
        fill_stamp(odometry.header, estimate.stamp_ns)
        odometry.header.frame_id = output.frame_id
        odometry.child_frame_id = self.bridge_config.child_frame_id
        position = odometry.pose.pose.position
        position.x, position.y, position.z = map(float, output.xyz)
        odometry.pose.pose.orientation.z = math.sin(output.yaw_rad/2)
        odometry.pose.pose.orientation.w = math.cos(output.yaw_rad/2)
        odometry.pose.covariance = list(output.pose_covariance)
        odometry.twist.twist.linear.x = float(estimate.speed_mps)
        covariance = [0.0]*36
        covariance[0] = max(0.0, float(estimate.covariance[1][1]))
        for i in (7, 14, 21, 28, 35):
            covariance[i] = 1e6  # The longitudinal model does not observe these components.
        odometry.twist.covariance = covariance
        self.velocity_pub.publish(velocity)
        self._record_publish_gap("velocity", estimate.stamp_ns)
        self.position_pub.publish(odometry)
        self._record_publish_gap("position", estimate.stamp_ns)
        self.drive_model_counts[output.diagnostics["drive_model_source"]] += 1
        elapsed_ms=(perf_counter()-began)*1000
        timing=self.publish_timing[output.diagnostics["trigger"]]
        timing["count"]+=1
        timing["total_ms"]+=elapsed_ms
        timing["max_ms"]=max(timing["max_ms"],elapsed_ms)
        timing["over_100ms_count"]+=int(elapsed_ms>100)
        self._sync_gnss_subscription()
        if estimate.stamp_ns-self.last_diagnostic_ns >= 1_000_000_000:
            self.last_diagnostic_ns = estimate.stamp_ns
            diagnostics = dict(output.diagnostics)
            diagnostics["position_reference_point"] = "master_gnss_projected_to_map"
            diagnostics["height_reference"] = "calibrated_map_profile_empirical_offset"
            diagnostics["surveyed_body_transform_confirmed"] = False
            diagnostics["child_frame_is_reference_alias"] = True
            diagnostics["initial_refinement_complete"] = self.bridge.fusion.refinement_closed
            diagnostics["late_position_tracking_enabled"] = False
            diagnostics["callback_work_ms"] = round(elapsed_ms, 3)
            diagnostics.update(self.timing_diagnostics())
            diagnostics["callback_work_is_end_to_end_latency"] = False
            array = DiagnosticArray()
            fill_stamp(array.header, estimate.stamp_ns)
            status = DiagnosticStatus()
            status.name, status.hardware_id = "tram_odometry", "wheel_odometry"
            degraded = (diagnostics["command_stale"] or not diagnostics["route_initialized"]
                        or diagnostics["slip_flag"] or diagnostics["outside_map_m"] > 0)
            status.level = DiagnosticStatus.WARN if degraded else DiagnosticStatus.OK
            status.message = "degraded / initialization or model fallback" if degraded else "estimating"
            status.values = [KeyValue(key=str(k), value=str(v)) for k, v in diagnostics.items()]
            array.status = [status]
            self.diagnostic_pub.publish(array)

    def _record_publish_gap(self, kind, stamp_ns):
        now = perf_counter()
        values = self.publish_gaps[kind]
        previous = values["last_at"]
        if previous is not None:
            gap_ms = (now-previous)*1000
            if gap_ms > values["max_ms"]:
                values["max_ms"] = gap_ms
                values["max_stamp_ns"] = stamp_ns
            values["over_100ms_count"] += int(gap_ms > 100)
        values["last_at"] = now


    def timing_diagnostics(self):
        result={"drive_model_tick_counts_json":json.dumps(self.drive_model_counts,sort_keys=True)}
        for kind, values in self.publish_timing.items():
            prefix=kind+"_callback_to_publish_"
            result[prefix+"count"]=values["count"]
            result[prefix+"mean_ms"]=round(values["total_ms"]/values["count"] if values["count"] else 0.0,3)
            result[prefix+"max_ms"]=round(values["max_ms"],3)
            result[prefix+"over_100ms_count"]=values["over_100ms_count"]
        for kind, values in self.publish_gaps.items():
            prefix=kind+"_internal_publish_gap_"
            result[prefix+"max_ms"]=round(values["max_ms"],3)
            result[prefix+"max_stamp_ns"]=values["max_stamp_ns"]
            result[prefix+"over_100ms_count"]=values["over_100ms_count"]
        return result

    def log_timing_summary(self):
        values=self.timing_diagnostics()
        print("[tram_odometry] Callback-to-publish totals (DDS delivery/queue wait excluded): "+
              ", ".join(str(k)+"="+str(v) for k,v in values.items()),flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OdometryNode()
        rclpy.spin(node)  # Single-threaded callbacks preserve receive order.
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            node.log_timing_summary()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
