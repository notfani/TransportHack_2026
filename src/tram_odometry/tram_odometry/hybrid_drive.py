"""ROS-side adapter for the delivered, standard-library-only acceleration model."""
from collections import deque
import json
import math
from pathlib import Path

from tram_estimator import Drive
from tram_hybrid_model.hybrid import HybridModel, HybridState, observe_trusted
from tram_hybrid_model.longitudinal import Params
from tram_hybrid_model.residual_export import predict_residual


class HybridDrive:
    def __init__(self, config_path):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        params = Params.from_mapping(cfg["physical"])
        self.model = HybridModel(
            params, predict_residual, uses_memory=cfg["uses_memory"],
            min_speed=cfg["min_speed"], max_speed=cfg["max_speed"])
        self.state = HybridState()
        self.history = deque(maxlen=11)
        self.last_sample_ns = 0
        self.source = "physics_startup"

    def reset_unavailable(self):
        self.state = HybridState()
        self.history.clear()
        self.last_sample_ns = 0
        self.source = "command_unavailable"

    def predict_interval(self, command, speed_mps, dt):
        if not math.isfinite(dt) or not 0 < dt <= self.model.p.max_dt:
            self.reset_unavailable()
            self.source = "invalid_interval"
            return Drive(0.0, 1.0)
        result, self.state, self.source = self.model.predict(
            command, speed_mps, dt, self.state)
        if not result.valid:
            return Drive(0.0, max(1.0, result.accel_uncertainty))
        return Drive(result.a_model, result.accel_uncertainty)

    def observe_wheels(self, estimate, front, rear):
        # Memory is derived only from two recently received, trusted wheels.
        if estimate.slip_flag:
            self.history.clear()
            self.last_sample_ns = 0
            return
        if estimate.trust_wheels < self.model.p.min_trust:
            # Alternating command ticks may have no newly consumed wheel sample.
            # They do not invalidate the trailing trusted 10 Hz history.
            return
        wheels = (front, rear)
        if any(w is None or not math.isfinite(w.speed_mps)
               or w.speed_mps < 0 or not 0 <= (estimate.stamp_ns-w.stamp_ns)/1e9 <= .2
               for w in wheels):
            self.history.clear()
            self.last_sample_ns = 0
            return
        if abs(front.stamp_ns-rear.stamp_ns) > 50_000_000:
            self.history.clear()
            self.last_sample_ns = 0
            return
        t = estimate.stamp_ns
        if self.last_sample_ns:
            dt = (t-self.last_sample_ns)/1e9
            if dt < .085:
                return
            if dt > .125:
                self.history.clear()
        self.history.append((t, (front.speed_mps+rear.speed_mps)/2))
        self.last_sample_ns = t
        if len(self.history) < 11:
            return
        times = [(stamp-self.history[5][0])/1e9 for stamp, _ in self.history]
        speeds = [speed for _, speed in self.history]
        mean_t = sum(times)/11
        mean_v = sum(speeds)/11
        denom = sum((x-mean_t)**2 for x in times)
        if denom <= 0:
            return
        accel = sum((x-mean_t)*(v-mean_v)
                    for x, v in zip(times, speeds))/denom
        if abs(accel) <= min(2.0, self.model.p.max_accel):
            self.state = observe_trusted(
                self.state, accel, trust_wheels=estimate.trust_wheels,
                slip=False, fresh=True, p=self.model.p)
