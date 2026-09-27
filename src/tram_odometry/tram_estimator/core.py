"""Causal robust longitudinal estimator with full [s, v, b_a] covariance."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def tr(a):
    return [list(row) for row in zip(*a)]


@dataclass(frozen=True)
class Wheel:
    stamp_ns: int
    speed_mps: float


@dataclass(frozen=True)
class Drive:
    acceleration_mps2: float
    sigma_mps2: float = 0.5


@dataclass(frozen=True)
class Config:
    max_speed_mps: float = 22.0
    min_accel_mps2: float = -2.5
    max_accel_mps2: float = 1.8
    wheel_age_s: float = 0.35
    wheel_sigma_mps: float = 0.06
    acceleration_density: float = 0.30
    model_error_correlation_s: float = 1.0
    bias_density: float = 0.002
    bias_limit_mps2: float = 0.8
    split_mps: float = 0.56
    innovation_mps: float = 0.55
    jump_mps2: float = 10.0
    model_only_s: float = 5.0
    recovery_history_s: float = 5.0
    stationary_mps: float = 0.05
    max_step_s: float = 0.10

    def __post_init__(self):
        for name in ("max_speed_mps", "wheel_age_s", "wheel_sigma_mps", "max_step_s",
                     "model_error_correlation_s", "innovation_mps", "split_mps", "recovery_history_s"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(name+" must be finite and positive")
        for name in ("acceleration_density", "bias_density", "bias_limit_mps2", "model_only_s"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(name+" must be finite and nonnegative")
        if not (math.isfinite(self.min_accel_mps2) and math.isfinite(self.max_accel_mps2)
                and self.min_accel_mps2<0<self.max_accel_mps2):
            raise ValueError("acceleration bounds must be finite and straddle zero")


@dataclass(frozen=True)
class Estimate:
    stamp_ns: int
    distance_m: float
    speed_mps: float
    acceleration_bias_mps2: float
    covariance: tuple[tuple[float, ...], ...]
    trust_wheels: float
    slip_flag: bool
    source: str
    reasons: tuple[str, ...]


@dataclass
class _Sensor:
    observed_stamp_ns: int = 0
    observed_speed_mps: float | None = None
    accepted_stamp_ns: int = 0
    suspect: bool = False
    suspect_since_ns: int = 0
    jump_sign: float = 0.0
    changed_stamp_ns: int = 0
    last_change_rate: float = 0.0
    return_until_ns: int = 0


class Estimator:
    """One call per output tick, supplied only with values already received.

    Every wheel (sensor, stamp) is consumed once. Delayed measurements within
    wheel_age_s are transported to the tick with a causal bounded slope and
    inflated variance. Older values are unavailable, never interpolated.
    """

    def __init__(self, config: Config | None = None, *, distance_m: float = 0.0):
        self.config = config or Config()
        self.x = [float(distance_m), 0.0, 0.0]
        self.p = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.25]]
        self.stamp_ns = 0
        self.sensors = {"front": _Sensor(), "rear": _Sensor()}
        self.shadow_speed = 0.0
        self.shadow_active = False
        self.shadow_since_ns = 0
        self.initialized_speed = False
        self.history = []  # [anchor_stamp, model v, frozen bias, original v]
        # Uncorrected onset trajectory survives a rolling-history recovery.
        # [stamp, predicted v, frozen bias, {sensor: jump sign}]; bounded lifetime.
        self.episode = None
        self.episode_close_since_ns = 0

    def _predict(self, dt: float, drive: Drive):
        c = self.config
        a0 = drive.acceleration_mps2 if math.isfinite(drive.acceleration_mps2) else 0.0
        a = clip(a0 + self.x[2], c.min_accel_mps2, c.max_accel_mps2)
        old_v = self.x[1]
        v = clip(old_v + a * dt, 0.0, c.max_speed_mps)
        self.x[0] += (old_v + v) * dt * 0.5
        self.x[1] = v
        if self.shadow_active:
            self.shadow_speed = clip(self.shadow_speed + a * dt, 0.0, c.max_speed_mps)
        if self.episode is not None:
            episode_a = clip(a0+self.episode[2],c.min_accel_mps2,c.max_accel_mps2)
            self.episode[1] = clip(self.episode[1]+episode_a*dt,0.0,c.max_speed_mps)
        for checkpoint in self.history:
            old_a = clip(a0+checkpoint[2],c.min_accel_mps2,c.max_accel_mps2)
            checkpoint[1] = clip(checkpoint[1]+old_a*dt,0.0,c.max_speed_mps)
        f = [[1.0, dt, 0.5*dt*dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]
        p = mm(mm(f, self.p), tr(f))
        qa = c.acceleration_density
        qb = c.bias_density
        sigma = drive.sigma_mps2 if math.isfinite(drive.sigma_mps2) else 0.5
        # sigma describes acceleration error, not a spectral density. The
        # explicit correlation time converts it to m²/s³ for this approximation.
        qa = max(qa, sigma*sigma*c.model_error_correlation_s)
        q = [
            [qa*dt**3/3+qb*dt**5/20, qa*dt**2/2+qb*dt**4/8, qb*dt**3/6],
            [qa*dt**2/2+qb*dt**4/8, qa*dt+qb*dt**3/3, qb*dt**2/2],
            [qb*dt**3/6, qb*dt**2/2, qb*dt],
        ]
        self.p = [[p[i][j]+q[i][j] for j in range(3)] for i in range(3)]

    def _correct(self, z: float, variance: float, weight: float, adapt: bool):
        if weight <= 0.0:
            return
        r = variance / (weight*weight)
        denom = self.p[1][1] + r
        k = [self.p[i][1] / denom for i in range(3)]
        if not adapt:
            k[2] = 0.0
        innovation = z - self.x[1]
        self.x = [self.x[i] + k[i]*innovation for i in range(3)]
        self.x[1] = clip(self.x[1], 0.0, self.config.max_speed_mps)
        self.x[2] = clip(self.x[2], -self.config.bias_limit_mps2, self.config.bias_limit_mps2)
        # Joseph form supports the deliberately frozen bias gain as well.
        a = [[float(i == j) - (k[i] if j == 1 else 0.0) for j in range(3)] for i in range(3)]
        p = mm(mm(a, self.p), tr(a))
        self.p = [[p[i][j] + k[i]*r*k[j] for j in range(3)] for i in range(3)]

    def update(self, stamp_ns: int, command: int, front: Wheel | None,
               rear: Wheel | None, drive: Drive) -> Estimate:
        if stamp_ns <= 0:
            raise ValueError("positive bag header timestamp required")
        if self.stamp_ns and stamp_ns <= self.stamp_ns:
            return self._result(0.0, "unchanged", ("non_increasing_tick",))
        c = self.config
        reasons = []
        initial = not self.stamp_ns
        if self.stamp_ns:
            remaining = (stamp_ns-self.stamp_ns)/1e9
            if remaining > 1.0:
                reasons.append("tick_gap")
            while remaining > 1e-9:
                dt = min(remaining, c.max_step_s)
                self._predict(dt, drive)
                remaining -= dt
        self.stamp_ns = stamp_ns
        if self.episode is not None and stamp_ns-self.episode[0]>30_000_000_000:
            self.episode = None
        a = clip((drive.acceleration_mps2 if math.isfinite(drive.acceleration_mps2) else 0.0)
                 + self.x[2], c.min_accel_mps2, c.max_accel_mps2)
        valid = {}
        fresh = {}
        for name, wheel in (("front", front), ("rear", rear)):
            sensor = self.sensors[name]
            if wheel is None:
                continue
            if wheel.stamp_ns < sensor.observed_stamp_ns:
                reasons.append(name+"_out_of_order")
                continue
            age = (stamp_ns-wheel.stamp_ns)/1e9
            if (not math.isfinite(wheel.speed_mps) or wheel.speed_mps < 0.0
                    or wheel.speed_mps > c.max_speed_mps*2):
                reasons.append(name+"_invalid")
                continue
            if age < 0.0 or age > c.wheel_age_s:
                reasons.append(name+"_stale_or_future")
                continue
            rate = None
            is_new = wheel.stamp_ns > sensor.observed_stamp_ns
            if is_new and sensor.observed_stamp_ns and sensor.observed_speed_mps is not None:
                dtw = (wheel.stamp_ns-sensor.observed_stamp_ns)/1e9
                if 0.02 <= dtw <= c.wheel_age_s*2:
                    rate = (wheel.speed_mps-sensor.observed_speed_mps)/dtw
            jump = rate is not None and abs(rate) > c.jump_mps2
            transport_a = clip(rate, c.min_accel_mps2, c.max_accel_mps2) if rate is not None and not jump else a
            z = max(0.0, wheel.speed_mps + transport_a*age)
            variance = c.wheel_sigma_mps**2 + (0.6*age)**2
            valid[name] = (z, variance, rate, jump)
            if is_new:
                changed = (sensor.observed_speed_mps is None
                           or abs(wheel.speed_mps-sensor.observed_speed_mps)>1e-6)
                plateau_s = (sensor.observed_stamp_ns-sensor.changed_stamp_ns)/1e9
                # Keep return evidence briefly for asynchronous bogie messages.
                # An opposite jump is informative only within an active episode.
                if jump:
                    sensor.return_until_ns = 0
                episode = self.episode
                reverse_jump = (jump and episode is not None and len(episode[3])==2
                                and rate*episode[3].get(name,0)<0
                                and abs(z-episode[1])+c.innovation_mps
                                < abs(sensor.observed_speed_mps-episode[1]))
                thaw = (jump and plateau_s>=.5 and sensor.observed_speed_mps>1.
                        and abs(sensor.last_change_rate)>.2)
                sigma = abs(drive.sigma_mps2) if math.isfinite(drive.sigma_mps2) else .5
                if reverse_jump:
                    elapsed = (stamp_ns-episode[0])/1e9
                    reverse_jump = abs(z-episode[1])<=c.innovation_mps+sigma*elapsed
                if thaw:
                    candidates = [h for h in self.history if h[0]<=sensor.changed_stamp_ns]
                    if candidates:
                        h = candidates[-1]
                        tolerance = c.innovation_mps+sigma*(stamp_ns-h[0])/1e9
                        thaw = (abs(z-h[1])<=tolerance and
                                abs(z-h[1])+c.innovation_mps<abs(self.x[1]-h[1]))
                    else:
                        thaw = False
                if reverse_jump or thaw:
                    sensor.return_until_ns = stamp_ns+350_000_000
                if changed:
                    sensor.changed_stamp_ns = wheel.stamp_ns
                    sensor.last_change_rate = rate or 0.
                fresh[name] = valid[name]
                sensor.observed_stamp_ns = wheel.stamp_ns
                sensor.observed_speed_mps = wheel.speed_mps
        if initial or not self.initialized_speed:
            if fresh:
                self.x[1] = clip(sum(z[0] for z in fresh.values())/len(fresh),0.0,c.max_speed_mps)
                # Before bootstrap, model propagation may have accumulated
                # cross terms. Replacing only Pvv would destroy PSD.
                for i in (0,2):
                    self.p[i][1] = self.p[1][i] = 0.0
                self.p[1][1] = c.wheel_sigma_mps**2
                self.shadow_speed = self.x[1]
                self.initialized_speed = True
                if len(valid)==2 and abs(valid["front"][0]-valid["rear"][0])>c.split_mps:
                    reasons.append("initial_bogie_split")
                    self.p[1][1] = max(self.p[1][1],(valid["front"][0]-valid["rear"][0])**2/4)
                    self._suspect("front",stamp_ns,None)
                    self._suspect("rear",stamp_ns,None)
            trust = 1.0 if fresh and not any(s.suspect for s in self.sensors.values()) else 0.0
            return self._result(trust, "initial", tuple(reasons))
        split = len(valid) == 2 and abs(valid["front"][0]-valid["rear"][0]) > c.split_mps
        if split:
            reasons.append("bogie_split")
        anchor = self.shadow_speed if self.shadow_active else self.x[1]
        gate = c.innovation_mps
        coherent_recovery = False
        if (len(valid)==2 and not split and self.history and fresh
                and all(s.return_until_ns>=stamp_ns for s in self.sensors.values())):
            # Both return transitions were checked against independent model
            # trajectories when received, before a first bogie could reset the
            # common shadow during the asynchronous second-bogie wait.
            coherent_recovery = True
            if coherent_recovery:
                for sensor in self.sensors.values():
                    sensor.suspect = False
                    sensor.return_until_ns = 0
                self.episode = None
                reasons.append("coherent_recovery")
        elif (len(valid)==2 and not split and self.history and fresh
              and (any(z[3] for z in fresh.values())
                   or any(s.suspect for s in self.sensors.values()))):
            # A ramp may have no detectable onset. Preserve the short-window
            # change-point test. The separate onset trajectory is not discarded
            # by this heuristic and remains available for a later return jump.
            hstamp, independent_v, _, original_v = self.history[0]
            horizon = (stamp_ns-hstamp)/1e9
            zmean = sum(z[0] for z in valid.values())/2
            sigma = abs(drive.sigma_mps2) if math.isfinite(drive.sigma_mps2) else .5
            preceding_change = anchor-original_v
            coherent_recovery = (horizon>=1.0 and abs(zmean-independent_v)<=gate+sigma*horizon
                                 and abs(zmean-independent_v)+gate<abs(anchor-independent_v)
                                 and abs(preceding_change)>gate
                                 and preceding_change*(zmean-anchor)<0)
            if coherent_recovery:
                for sensor in self.sensors.values():
                    sensor.suspect = False
                reasons.append("coherent_recovery")
        selected = dict(fresh)
        if split and fresh:
            closest = min(valid, key=lambda name: abs(valid[name][0]-anchor))
            selected = {closest: fresh[closest]} if closest in fresh else {}
            for name in fresh:
                if name != closest or abs(fresh[name][0]-anchor) > gate:
                    self._suspect(name, stamp_ns, fresh[name][2])
        for name, (z, variance, rate, jump) in fresh.items():
            if jump and abs(z-anchor) > gate and not coherent_recovery:
                self._suspect(name, stamp_ns, rate)
                reasons.append(name+"_jump")
            if rate is not None and abs(rate-a) > 2.2:
                reasons.append(name+"_accel_residual")
            if command > 0 and rate is not None and rate > a+2.2:
                reasons.append("traction_spin")
            if command < 0 and rate is not None and rate < a-2.2:
                reasons.append("brake_slide")
            if abs(command) >= 8 and abs(a) > 0.4 and rate is not None and abs(rate) < 0.05:
                reasons.append("stalled_response")
            if self.sensors[name].suspect and abs(z-anchor) <= gate:
                self.sensors[name].suspect = False
                reasons.append(name+"_recovered")
        if any(s.suspect for s in self.sensors.values()) and not self.shadow_active:
            self.shadow_active = True
            self.shadow_speed = self.x[1]
            self.shadow_since_ns = stamp_ns
        weights = []
        used = []
        both_healthy = len(valid) == 2 and not split and not any(s.suspect for s in self.sensors.values())
        for name, (z, variance, rate, jump) in selected.items():
            sensor = self.sensors[name]
            innovation = abs(z-self.x[1])
            weight = 1.0
            if sensor.suspect:
                elapsed = (stamp_ns-sensor.suspect_since_ns)/1e9
                weight = 0.0 if elapsed <= c.model_only_s else 0.05
                reasons.append(name+"_suspect")
            elif split or len(valid) == 1:
                last_accepted = max(s.accepted_stamp_ns for s in self.sensors.values())
                reacquiring = (not split and len(valid)==1
                               and (stamp_ns-last_accepted)/1e9>c.wheel_age_s)
                adaptive_gate = max(gate,3*math.sqrt(max(0.0,self.p[1][1])+variance)) if reacquiring else gate
                weight = (0.4 if reacquiring else 0.8) if innovation <= adaptive_gate and not jump else 0.0
                if reacquiring and weight:
                    reasons.append("single_reacquisition")
                if weight == 0.0:
                    reasons.append(name+"_innovation")
            if weight:
                # Two bogies share systematic error: paired measurements do not
                # claim a sqrt(2) reduction below one wheel's noise floor.
                self._correct(z, variance*(2 if len(selected) == 2 else 1), weight,
                              both_healthy and not coherent_recovery)
                sensor.accepted_stamp_ns = stamp_ns
                used.append(name)
            weights.append(weight)
        if both_healthy and fresh:
            self.shadow_active = False
            self.shadow_speed = self.x[1]
        if both_healthy and fresh and max(z[0] for z in valid.values()) < c.stationary_mps and self.x[1] < 0.25:
            self.x[1] = 0.0
            # Do not learn a fictitious mass/drive response from a clamped stop.
            self.x[2] *= 0.99
        if self.episode is not None and fresh:
            returned = (len(valid)==2 and not split and not any(z[3] for z in fresh.values())
                        and all(abs(z[0]-self.episode[1])<=gate for z in valid.values()))
            if returned:
                if not self.episode_close_since_ns:
                    self.episode_close_since_ns = stamp_ns
                elif stamp_ns-self.episode_close_since_ns>=500_000_000:
                    self.episode = None
            else:
                self.episode_close_since_ns = 0
        elif self.episode is not None and len(valid)<2:
            self.episode_close_since_ns = 0
        if self.episode is None:
            self.episode_close_since_ns = 0
        source = "wheels" if len(used) == 2 else used[0] if used else "model"
        return self._result(max(weights, default=0.0), source, tuple(dict.fromkeys(reasons)))

    def _suspect(self, name, stamp_ns, rate):
        sensor = self.sensors[name]
        if rate is not None and abs(rate)>self.config.jump_mps2:
            if self.episode is None or (len(self.episode[3])<2 and stamp_ns-self.episode[0]>350_000_000):
                self.episode = [stamp_ns,self.x[1],self.x[2],{}]
                self.episode_close_since_ns = 0
            if stamp_ns-self.episode[0]<=350_000_000:
                signs = self.episode[3]
                sign = math.copysign(1.,rate)
                if not signs or sign==next(iter(signs.values())):
                    signs[name] = sign
        if not sensor.suspect:
            sensor.suspect = True
            sensor.suspect_since_ns = stamp_ns
            sensor.jump_sign = math.copysign(1.0, rate) if rate else 0.0

    def _result(self, trust, source, reasons):
        self.history = [h for h in self.history if self.stamp_ns-h[0]<=self.config.recovery_history_s*1e9]
        if self.initialized_speed and (not self.history or self.stamp_ns-self.history[-1][0]>=500_000_000):
            self.history.append([self.stamp_ns,self.x[1],self.x[2],self.x[1]])
        return Estimate(self.stamp_ns, self.x[0], self.x[1], self.x[2],
                        tuple(tuple(row) for row in self.p), trust,
                        any(s.suspect for s in self.sensors.values()), source, reasons)
