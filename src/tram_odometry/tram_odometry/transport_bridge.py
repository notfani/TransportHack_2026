"""Causal transport, units, manual modes and watchdog; no GNSS correction owner."""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, fields
import math

from tram_estimator import Config, Drive, Estimator, Wheel
from tram_position.route_localizer import Anchor


@dataclass(frozen=True)
class BridgeConfig:
    wheel_input_unit: str = "kmh"
    watchdog_after_s: float = 0.06
    watchdog_period_s: float = 0.02
    command_timeout_s: float = 0.2
    missing_command_sigma_mps2: float = 1.0
    initialization_window_s: float = 3.0
    initial_fix_tolerance_s: float = 0.15
    initial_heading_min_displacement_m: float = 3.0
    max_initial_lateral_m: float = 30.0
    start_mode: str = "gnss"  # gnss | route_chainage | map_pose | relative
    route_name: str = "auto"
    start_chainage_m: float = 0.0
    start_x_m: float = 0.0
    start_y_m: float = 0.0
    start_z_m: float = 0.0
    start_yaw_rad: float = 0.0
    map_frame_id: str = "pathgraph"
    relative_frame_id: str = "odom_relative"
    child_frame_id: str = "base_link"
    output_yaw_rad: float = 0.0
    output_x_m: float = 0.0
    output_y_m: float = 0.0
    output_z_m: float = 0.0
    map_sigma_xy_m: float = 3.0
    map_sigma_z_m: float = 5.0
    map_sigma_yaw_rad: float = 0.2


@dataclass(frozen=True)
class Output:
    estimate: object
    xyz: tuple[float, float, float]
    yaw_rad: float
    frame_id: str
    pose_covariance: tuple[float, ...]
    diagnostics: dict


class BridgeBase:
    def __init__(self, drive_model, localizer, config=None, estimator_config=None):
        self.c = config or BridgeConfig()
        c = self.c
        if any(isinstance(getattr(c, f.name), float) and not math.isfinite(getattr(c, f.name)) for f in fields(c)):
            raise ValueError("all numeric adapter parameters must be finite")
        if c.wheel_input_unit not in ("kmh", "mps"):
            raise ValueError("wheel_input_unit must explicitly be kmh or mps")
        if c.start_mode not in ("gnss", "route_chainage", "map_pose", "relative"):
            raise ValueError("unknown start_mode")
        if min(c.watchdog_after_s, c.watchdog_period_s, c.command_timeout_s) <= 0:
            raise ValueError("watchdog and command timeout must be positive")
        if c.watchdog_after_s+c.watchdog_period_s > 0.1:
            raise ValueError("watchdog nominal maximum output gap must not exceed 0.1s")
        if not 0 <= c.initialization_window_s <= 5.0:
            raise ValueError("GNSS initialization window must be between 0 and 5 seconds")
        if min(c.map_sigma_xy_m, c.map_sigma_z_m, c.map_sigma_yaw_rad) < 0:
            raise ValueError("uncertainty floors must be nonnegative")
        self.estimator = Estimator(estimator_config or Config())
        self.drive_model, self.localizer = drive_model, localizer
        self.front = self.rear = None
        self.command, self.command_stamp_ns = 0, 0
        self.first_input_ns = self.clock_anchor_ns = self.header_anchor_ns = 0
        self.last_command_clock_ns = 0
        # A finite grace period lets delayed headers catch a watchdog forecast.
        # A permanently lagging command stream must eventually fall back to
        # model-only watchdog outputs instead of freezing state time forever.
        self.late_command_catchup_started_ns = 0
        self.last_output = None
        self.history = deque(maxlen=1000)
        self.fixes = deque(maxlen=100)
        # Failed projections are pure functions of initial position and heading.
        # Repeating every old fix at every callback used O(fixes * map) work.
        # Cache exact inputs only: a newly available heading must still retry.
        self.failed_anchor_attempts = OrderedDict()
        self.anchor = None
        self.anchor_distance_m = 0.0
        self.initialization_reason = "awaiting_initial_fix" if c.start_mode == "gnss" else c.start_mode
        self.gnss_closed = c.start_mode != "gnss" or c.initialization_window_s <= 0
        self.counters = {"out_of_order_command": 0, "out_of_order_wheel": 0,
                         "invalid_input": 0, "ignored_gnss": 0}
        self._manual_anchor()

    def _manual_anchor(self):
        c = self.c
        if c.start_mode not in ("route_chainage", "map_pose"):
            return
        routes = [r for r in self.localizer.routes if r.name == c.route_name]
        if len(routes) != 1:
            raise ValueError("manual map initialization requires an explicit valid route_name")
        route = routes[0]
        if c.start_mode == "route_chainage":
            if not 0 <= c.start_chainage_m <= route.length_m:
                raise ValueError("start_chainage_m outside configured route")
            self.anchor = Anchor(route.name, c.start_chainage_m, 0.0, "configured_chainage")
        else:
            s, lateral = route.project(c.start_x_m, c.start_y_m)
            if lateral > c.max_initial_lateral_m:
                raise ValueError("configured start pose too far from route")
            self.anchor = Anchor(route.name, s, lateral, "configured_map_pose")

    def _note_first_input(self, stamp_ns, clock_ns):
        if not self.first_input_ns:
            self.first_input_ns = stamp_ns
            self.header_anchor_ns, self.clock_anchor_ns = stamp_ns, clock_ns

    def receive_wheel(self, name, stamp_ns, speed, clock_ns):
        if name not in ("front", "rear"):
            raise ValueError("unknown wheel")
        if stamp_ns <= 0 or not math.isfinite(speed) or speed < 0 or speed/(3.6 if self.c.wheel_input_unit == "kmh" else 1.0) > 2*self.estimator.config.max_speed_mps:
            self.counters["invalid_input"] += 1
            return
        previous = getattr(self, name)
        if previous is not None and stamp_ns <= previous.stamp_ns:
            self.counters["out_of_order_wheel"] += 1
            return
        self._note_first_input(stamp_ns, clock_ns)
        setattr(self, name, Wheel(stamp_ns, speed/(3.6 if self.c.wheel_input_unit == "kmh" else 1.0)))

    def receive_command(self, stamp_ns, command, clock_ns):
        if stamp_ns <= 0 or not -15 <= command <= 15:
            self.counters["invalid_input"] += 1
            return None
        if self.command_stamp_ns and stamp_ns <= self.command_stamp_ns:
            self.counters["out_of_order_command"] += 1
            return None
        self._note_first_input(stamp_ns, clock_ns)
        # The NEW command cannot influence the interval that ended at its stamp.
        # A receive-time pause can let the watchdog project past a command's
        # header stamp. Keep the state clock monotone, but publish a tiny
        # recovery step for each newly arrived command while its header catches
        # up. Otherwise _advance returns None forever, the command receipt
        # clock is never refreshed, and the watchdog keeps racing ahead.
        late = self.estimator.stamp_ns and stamp_ns <= self.estimator.stamp_ns
        if late and not self.late_command_catchup_started_ns:
            self.late_command_catchup_started_ns = clock_ns
        if not late:
            self.late_command_catchup_started_ns = 0
        catching_up = (late and clock_ns-self.late_command_catchup_started_ns <= 1_500_000_000)
        target_ns = self.estimator.stamp_ns + 1 if catching_up else stamp_ns
        output = self._advance(target_ns, "command")
        self.command, self.command_stamp_ns = command, stamp_ns
        if output is not None:
            self.last_command_clock_ns = clock_ns
        # Do not move the output clock backwards after a slightly late command.
        if output is not None:
            self.header_anchor_ns, self.clock_anchor_ns = self.estimator.stamp_ns, clock_ns
        return output

    def tick(self, clock_ns):
        if not self.first_input_ns or clock_ns <= 0 or clock_ns < self.clock_anchor_ns:
            return None
        last_command = self.last_command_clock_ns or self.clock_anchor_ns
        if (clock_ns-last_command)/1e9 < self.c.watchdog_after_s:
            return None
        # /clock carries bag RECEIVE time, which can differ from header by seconds.
        # Advance the last input-header anchor by elapsed replay time instead.
        stamp_ns = self.header_anchor_ns + (clock_ns-self.clock_anchor_ns)
        return self._advance(stamp_ns, "watchdog")


