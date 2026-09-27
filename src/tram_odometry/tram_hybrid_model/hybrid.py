"""Causal ML residual over physical dynamics. No ROS/NumPy/CatBoost dependency."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, isfinite
from typing import Callable

from .longitudinal import AdaptiveState, DriveEstimate, Params, clamp, predict, update_params

TIME_CONSTANTS = (0.2, 1.0, 3.0)
FEATURE_NAMES = ("u", "v", "u_ema_0.2", "u_ema_1", "u_ema_3", "last_trusted_accel", "trusted_age")


@dataclass(frozen=True)
class HybridState:
    commands: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initialized: bool = False
    warmup: float = 0.0
    last_accel: float = 0.0
    trusted_age: float = 6.0
    adaptive: AdaptiveState = AdaptiveState()

    def __post_init__(self):
        if (
            len(self.commands) != 3
            or not all(isfinite(x) and -15 <= x <= 15 for x in self.commands)
            or not all(isfinite(x) for x in (self.warmup, self.last_accel, self.trusted_age))
            or self.warmup < 0
            or self.trusted_age < 0
            or abs(self.last_accel) > 2
        ):
            raise ValueError("Invalid hybrid state")


def command_step(u: float, dt: float, previous: HybridState) -> tuple[float, float, float]:
    if not previous.initialized:
        return (u, u, u)
    return tuple(
        u + (old - u) * exp(-dt / tau) for old, tau in zip(previous.commands, TIME_CONSTANTS)
    )


def observe_trusted(
    state: HybridState,
    acceleration: float,
    *,
    trust_wheels: float,
    slip: bool,
    fresh: bool,
    p: Params,
    interval_estimate: DriveEstimate | None = None,
    update_dt: float = 0.1,
) -> HybridState:
    """Refresh memory only from a trailing, causal trusted-wheel derivative.

    Optional gain update requires an estimate averaged over the SAME derivative
    interval. Omit it rather than mixing an instantaneous prediction and a lagged
    derivative. Use matching derivative weights, not an arbitrary mean.
    update_dt is elapsed time since the last adaptation tick (normally 0.1 s),
    NOT the one-second derivative window. The memory is used on the next prediction.
    """
    if (
        slip
        or not fresh
        or not isfinite(trust_wheels)
        or not p.min_trust <= trust_wheels <= 1
        or not isfinite(acceleration)
        or abs(acceleration) > min(p.max_accel, 2)
    ):
        return state
    adaptive = state.adaptive
    if interval_estimate is not None:
        adaptive = update_params(
            adaptive,
            interval_estimate,
            acceleration,
            update_dt,
            trust_wheels=trust_wheels,
            slip=slip,
            fresh=fresh,
            p=p,
        )
    return replace(state, last_accel=acceleration, trusted_age=0.0, adaptive=adaptive)


class HybridModel:
    """Instantiate once with the exported predict_residual and physical params.

    predict returns (DriveEstimate, new HybridState, source). Caller owns velocity
    integration and trust; no wheel/GNSS measurement is accepted by predict.
    Integrate as clamp(v + dt * a_model, 0, p.max_speed), including at standstill:
    floating-point cancellation at a speed bound can otherwise leave v < 0.
    """

    def __init__(
        self,
        p: Params,
        residual: Callable[[list[float]], float],
        *,
        uses_memory: bool = True,
        min_speed: float = 0.5,
        max_speed: float = 15.0,
    ):
        if p.drive_table:
            raise ValueError("Hybrid model requires its fitted analytic physical parameters")
        if not (
            isfinite(min_speed)
            and isfinite(max_speed)
            and 0 <= min_speed < max_speed <= p.max_speed
        ):
            raise ValueError("Invalid learned speed domain")
        self.p, self.residual = p, residual
        self.uses_memory = uses_memory
        self.min_speed, self.max_speed = min_speed, max_speed

    def predict(
        self, u: float, v: float, dt: float, state: HybridState = HybridState()
    ) -> tuple[DriveEstimate, HybridState, str]:
        base = predict(u, v, dt, self.p)
        if not base.valid:
            return base, HybridState(adaptive=state.adaptive), "invalid_input"
        commands = command_step(u, dt, state)
        new_state = replace(
            state,
            commands=commands,
            initialized=True,
            warmup=min(state.warmup + dt, 3.0),
            trusted_age=min(state.trusted_age + dt, 6.0),
        )
        weight = 1.0
        source = "hybrid"
        if state.warmup < 3 or not self.min_speed <= v <= self.max_speed:
            weight, source = 0.0, "physics_out_of_support"
        elif self.uses_memory and state.trusted_age > 5:
            weight = max(0.0, 6 - state.trusted_age)
            source = "fading_memory" if weight else "physics_stale_memory"
        features = [u, v, *commands]
        if self.uses_memory:
            features += [state.last_accel, min(state.trusted_age, 5.0)]
        correction = float(self.residual(features)) if weight else 0.0
        if not isfinite(correction):
            return (
                replace(base, valid=False, accel_uncertainty=self.p.max_accel),
                new_state,
                "invalid_ml",
            )
        nominal = base.nominal_drive_accel + weight * correction
        force_limit = min(self.p.max_force_n, self.p.adhesion_cap * self.p.mass_kg * 9.81)
        requested = (
            self.p.mass_kg
            * nominal
            * clamp(state.adaptive.drive_scale, self.p.scale_min, self.p.scale_max)
        )
        force = clamp(requested, -force_limit, force_limit)
        acceleration = force / self.p.mass_kg - base.resistance_accel
        limit = min(self.p.max_accel, self.p.max_force_n / self.p.mass_kg)
        # Analytic speed bounds; caller still projects after floating-point integration.
        bounded = clamp(acceleration, max(-limit, -v / dt), min(limit, (self.p.max_speed - v) / dt))
        if v == 0 and u <= 0:
            bounded = 0.0
        uncertainty = max(self.p.accel_uncertainty, 0.2 + 0.12 * state.trusted_age)
        if weight == 0:
            uncertainty = max(uncertainty, 1.0)
        d = DriveEstimate(
            bounded,
            force,
            self.p.mass_kg * bounded,
            base.resistance_accel,
            nominal,
            uncertainty,
            True,
            bounded != acceleration or force != requested,
        )
        return d, new_state, source
