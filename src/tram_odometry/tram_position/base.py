"""Bounded causal initialization; never changes the speed estimator."""
from __future__ import annotations
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field, replace
import math

@dataclass(frozen=True)
class InitializerConfig:
    route_name: str | None = None
    history_s: float = 10.0
    history_max: int = 2500
    pending_max: int = 400
    dedup_max: int = 2000
    initial_window_s: float = 3.0
    initial_tolerance_s: float = 0.15
    initial_heading_min_displacement_m: float = 3.0
    position_sigma_m: float = 3.0
    velocity_sigma_mps: float = 0.15
    initial_max_lateral_m: float = 30.0
    # This package has no tracking, speed-correction or calibration mode.
    mode: str = field(default='initialization_only', init=False)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
                raise ValueError(name + ' must be finite and nonnegative')
        if not 0 < self.history_s <= 10:
            raise ValueError('history must be bounded to at most ten seconds')
        for name in ('history_max', 'pending_max', 'dedup_max'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if not 0 <= self.initial_window_s <= 5:
            raise ValueError('initial refinement window must be in [0, 5] seconds')
        if min(self.position_sigma_m, self.velocity_sigma_mps, self.initial_heading_min_displacement_m) <= 0:
            raise ValueError('noise floors and heading displacement must be positive')

@dataclass(frozen=True)
class FusionOutput:
    estimate: object
    anchor: object | None
    chainage_m: float | None
    pose: object | None
    diagnostics: dict

@dataclass(frozen=True)
class _History:
    stamp: int
    distance: float
    speed: float
    variance_v: float
    raw_speed: float | None
    healthy: bool

class InitializerBase:
    def __init__(self, localizer, config=None, estimator=None):
        self.c = config or InitializerConfig()
        self.localizer, self.estimator = localizer, estimator
        self.history = deque(maxlen=self.c.history_max)
        self.pending = deque(maxlen=self.c.pending_max)
        self.seen = OrderedDict()
        self.counts = Counter()
        self.anchor = None
        self.offset_m = 0.0
        self.scale = 1.0
        self.first_stamp_ns = 0
        self.last_stamp_ns = 0
        self.last_accepted_fix_stamp_ns = 0
        self.last_observed_fix_stamp_ns = 0
        self.reacquire_allowed_until_ns = 0
        self.last_accepted_velocity_stamp_ns = 0
        self.odom_distance_at_correction = 0.0
        self.last_fix_xyz = None
        self.last_velocity = None
        self.initial_fixes = deque(maxlen=30)
        self.reacquire = deque(maxlen=10)
        self.wheel_cache = {}
        self.raw_wheels = {}
        self.scale_samples = 0
        self.scale_first_stamp_ns = 0
        self.scale_first_distance_m = 0.0
        self.scale_xx = self.scale_xy = 0.0
        self.last_scale_stamp_ns = 0
        self.last_innovation_m = None
        self.last_reason = "awaiting_fix"
        self.position_correction_m = 0.0
        self.accepted_this_tick = []
        self.reacquired_this_tick = False

    def _reject(self, kind, source, reason):
        self.counts[f"rejected_{kind}_{reason}"] += 1
        self.counts[f"rejected_{kind}_source_{source}"] += 1
        self.last_reason = reason

    def _queue(self, kind, stamp, source, payload):
        if source not in {"master", "rover"}:
            self._reject(kind, source, "unknown_source")
            return
        if not isinstance(stamp, int) or stamp <= 0:
            self._reject(kind, source, "invalid_stamp")
            return
        key = (kind, source, stamp)
        if key in self.seen:
            self._reject(kind, source, "duplicate")
            return
        self.seen[key] = None
        while len(self.seen) > self.c.dedup_max:
            self.seen.popitem(last=False)
        if len(self.pending) == self.pending.maxlen:
            old = self.pending.popleft()
            self._reject(old[0], old[2], "queue_overflow")
        self.pending.append((kind, stamp, source, payload))
        self.counts[f"received_{kind}"] += 1

    def receive_fix(self, stamp_ns, lat, lon, alt, status=0, source="master",
                    covariance_type=0, variance_m2=None):
        if (status < 0 or not all(math.isfinite(v) for v in (lat, lon, alt))
                or not -90 <= lat <= 90 or not -180 <= lon <= 180):
            self._reject("fix", source, "invalid")
            return
        variance = self.c.position_sigma_m**2
        if covariance_type and variance_m2 is not None:
            if not math.isfinite(variance_m2) or variance_m2 < 0:
                self._reject("fix", source, "invalid_covariance")
                return
            variance = max(variance, variance_m2)
        self._queue("fix", stamp_ns, source, (lat, lon, alt, variance))

    def receive_velocity(self, stamp_ns, vx, vy, vz, source="master", variance_m2ps2=None):
        if not all(math.isfinite(v) for v in (vx, vy, vz)):
            self._reject("velocity", source, "invalid")
            return
        speed = math.sqrt(vx*vx+vy*vy+vz*vz)
        if speed > 22:
            self._reject("velocity", source, "range")
            return
        variance = self.c.velocity_sigma_mps**2
        if variance_m2ps2 is not None:
            if not math.isfinite(variance_m2ps2) or variance_m2ps2 < 0:
                self._reject("velocity", source, "invalid_covariance")
                return
            variance = max(variance, variance_m2ps2)
        self._queue("velocity", stamp_ns, source, (speed, variance))

    def _raw_context(self, estimate, front, rear):
        values = []
        for name, wheel in (("front", front), ("rear", rear)):
            if wheel is None or not math.isfinite(wheel.speed_mps) or not 0 <= wheel.speed_mps <= 44:
                continue
            age = (estimate.stamp_ns-wheel.stamp_ns)/1e9
            if not 0 <= age <= .35:
                continue
            previous = self.raw_wheels.get(name)
            rate = previous[2] if previous is not None else 0.0
            if previous is not None and wheel.stamp_ns > previous[0]:
                dt = (wheel.stamp_ns-previous[0])/1e9
                if .02 <= dt <= .7:
                    rate = min(1.8, max(-2.5, (wheel.speed_mps-previous[1])/dt))
            if previous is None or wheel.stamp_ns > previous[0]:
                self.raw_wheels[name] = (wheel.stamp_ns, wheel.speed_mps, rate)
            values.append(max(0.0, wheel.speed_mps+rate*age))
        healthy = (len(values) == 2 and abs(values[0]-values[1]) <= .3
                   and not estimate.slip_flag and estimate.trust_wheels >= .8)
        return (sum(values)/len(values) if values else None), healthy

    def _at(self, stamp):
        if not self.history or stamp > self.history[-1].stamp:
            return None
        if stamp < self.history[0].stamp:
            if self.history[0].stamp == self.first_stamp_ns and self.first_stamp_ns-stamp <= self.c.initial_tolerance_s*1e9:
                return replace(self.history[0], stamp=stamp)
            return None
        previous = self.history[0]
        for current in self.history:
            if current.stamp == stamp:
                return current
            if current.stamp > stamp:
                f = (stamp-previous.stamp)/(current.stamp-previous.stamp)
                raw = (previous.raw_speed+f*(current.raw_speed-previous.raw_speed)
                       if previous.raw_speed is not None and current.raw_speed is not None else None)
                return _History(stamp,
                    previous.distance+f*(current.distance-previous.distance),
                    previous.speed+f*(current.speed-previous.speed),
                    max(previous.variance_v, current.variance_v), raw,
                    previous.healthy and current.healthy)
            previous = current
        return previous

    def _initial_anchor(self, stamp, source, payload, state):
        xyz = self.localizer.calibration.gnss_to_map(*payload[:3])
        self.initial_fixes.append((stamp, xyz, state.distance))
        while self.initial_fixes and stamp-self.initial_fixes[0][0] > self.c.history_s*1e9:
            self.initial_fixes.popleft()
        heading = None
        for first_stamp, first_xyz, distance in self.initial_fixes:
            dt = (stamp-first_stamp)/1e9
            displacement = math.hypot(xyz[0]-first_xyz[0], xyz[1]-first_xyz[1])
            if dt > 0 and self.c.initial_heading_min_displacement_m <= displacement <= 30*dt and state.distance-distance >= self.c.initial_heading_min_displacement_m:
                heading = math.atan2(xyz[1]-first_xyz[1], xyz[0]-first_xyz[0])+self.localizer.calibration.yaw_map_to_enu_rad
                break
        try:
            anchor = self.localizer.anchor(*payload[:3], route_name=self.c.route_name,
                heading_enu_rad=heading, max_lateral_m=self.c.initial_max_lateral_m)
        except ValueError:
            self._reject("fix", source, "initial_ambiguous_or_off_map")
            return
        self.anchor = anchor
        self.offset_m = anchor.chainage_m-state.distance
        self._accept_fix(stamp, source, xyz, state, "initial_"+anchor.selection_reason)
        self.initial_fixes.clear()

    def _accept_fix(self, stamp, source, xyz, state, reason):
        self.last_accepted_fix_stamp_ns = stamp
        self.odom_distance_at_correction = state.distance
        self.last_fix_xyz = xyz
        self.counts["accepted_fix"] += 1
        self.counts["accepted_fix_source_"+source] += 1
        self.last_reason = reason
        self.accepted_this_tick.append({"kind": "fix", "source": source, "stamp_ns": stamp,
            "innovation_m": None if reason.startswith("initial_") else self.last_innovation_m,
            "reason": reason})

    def _endpoint_outside(self, route, s, xyz):
        if s > 1e-6 and s < route.length_m-1e-6:
            return False
        a, b = route.points[:2] if s <= 1e-6 else route.points[-2:]
        origin = a if s <= 1e-6 else b
        dx, dy = b[0]-a[0], b[1]-a[1]
        along = ((xyz[0]-origin[0])*dx+(xyz[1]-origin[1])*dy)/max(1e-9, math.hypot(dx, dy))
        return along < -.5 if s <= 1e-6 else along > .5

    def observe(self, estimate, raw_front=None, raw_rear=None):
        if self.last_stamp_ns and estimate.stamp_ns <= self.last_stamp_ns:
            raise ValueError("observe needs a new immutable output tick")
        if not self.first_stamp_ns:
            self.first_stamp_ns = estimate.stamp_ns
        self.accepted_this_tick = []
        self.reacquired_this_tick = False
        self.last_stamp_ns = estimate.stamp_ns
        raw, healthy = self._raw_context(estimate, raw_front, raw_rear)
        self.history.append(_History(estimate.stamp_ns, estimate.distance_m, estimate.speed_mps,
                                     estimate.covariance[1][1], raw, healthy))
        threshold = estimate.stamp_ns-self.c.history_s*1e9
        while len(self.history) > 1 and self.history[0].stamp < threshold:
            self.history.popleft()
        ready, future = [], deque(maxlen=self.c.pending_max)
        for item in self.pending:
            if item[1] <= estimate.stamp_ns:
                ready.append(item)
            elif item[1]-estimate.stamp_ns <= self.c.history_s*1e9:
                future.append(item)
            else:
                self._reject(item[0], item[2], "future_too_far")
        self.pending = future
        # Preserve receive order. Same-stamp measurements from both receivers
        # cannot count as independent position/speed observations.
        for kind, stamp, source, payload in ready:
            state = self._at(stamp)
            if state is None:
                self._reject(kind, source, "stale")
                continue
            if kind == "fix":
                self._position(stamp, source, payload, state)
            else:
                estimate = self._velocity(stamp, source, payload, state, estimate)
        chainage, pose = None, None
        outside = 0.0
        if self.anchor is not None:
            chainage = estimate.distance_m+self.offset_m
            mapped = self.localizer.pose_after(self.anchor, chainage-self.anchor.chainage_m)
            outside = mapped.distance_outside_map_m
            if not outside:
                pose = mapped
        diagnostics = dict(self.counts)
        diagnostics.update(mode=self.c.mode, scale=self.scale, scale_learning_samples=self.scale_samples,
            last_accepted_fix_stamp_ns=self.last_accepted_fix_stamp_ns,
            last_accepted_velocity_stamp_ns=self.last_accepted_velocity_stamp_ns,
            odom_distance_at_correction=self.odom_distance_at_correction,
            correction_distance_m=self.position_correction_m,
            pre_correction_innovation_m=self.last_innovation_m,
            time_since_correction_s=((estimate.stamp_ns-self.last_accepted_fix_stamp_ns)/1e9
                                     if self.last_accepted_fix_stamp_ns else None),
            distance_since_correction_m=(abs(estimate.distance_m-self.odom_distance_at_correction)
                                        if self.last_accepted_fix_stamp_ns else None),
            absolute_valid=pose is not None, outside_map_m=outside,
            position_reason=self.last_reason, pending_count=len(self.pending), history_count=len(self.history),
            dedup_count=len(self.seen), covariance_calibrated=False,
            accepted_measurements=tuple(self.accepted_this_tick),
            reacquired_after_observed_gap=self.reacquired_this_tick,
            initialization_assumption=(self.anchor.selection_reason if self.anchor is not None else None))
        return FusionOutput(estimate, self.anchor, chainage, pose, diagnostics)

    def correct_wheel(self, name, wheel):
        return wheel

    def _position(self, stamp, source, payload, state):
        self.last_observed_fix_stamp_ns = max(self.last_observed_fix_stamp_ns, stamp)
        if self.anchor is None:
            self._initial_anchor(stamp, source, payload, state)
        else:
            self._reject('fix', source, 'initialization_complete')

    def _velocity(self, stamp, source, payload, state, estimate):
        self._reject('velocity', source, 'velocity_disabled')
        return estimate
