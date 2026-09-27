"""Pure longitudinal predictor and bounded adaptation; Python standard library only."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, fields
from math import isfinite


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


@dataclass(frozen=True)
class Params:
    mass_kg: float = 24000.0
    traction_accel: float = 1.2
    traction_notch: float = 10.0
    traction_exponent: float = 1.0
    brake_accel: float = 1.3
    brake_exponent: float = 0.7
    specific_power: float = 12.0  # P_effective / m, m²/s³; not motor nameplate power
    rolling_accel: float = 0.025
    linear_drag: float = 0.0
    quadratic_drag: float = 0.0002
    max_force_n: float = 48000.0
    adhesion_cap: float = 0.2  # Upper bound, NOT measured adhesion
    max_accel: float = 2.0
    max_speed: float = 25.0
    max_dt: float = 0.5
    accel_uncertainty: float = 0.35  # Engineering scale, not calibrated confidence
    scale_min: float = 0.65
    scale_max: float = 1.35
    adaptation_rate: float = 0.08  # 1/s
    adaptation_max_step_per_s: float = 0.025
    min_trust: float = 0.9
    speeds: tuple[float, ...] = ()
    drive_table: tuple[tuple[float, ...], ...] = ()  # 31 rows: notch -15 .. +15, m/s²

    def __post_init__(self):
        positive = (
            "mass_kg",
            "traction_accel",
            "traction_notch",
            "traction_exponent",
            "brake_accel",
            "brake_exponent",
            "specific_power",
            "max_force_n",
            "adhesion_cap",
            "max_accel",
            "max_speed",
            "max_dt",
            "accel_uncertainty",
            "scale_min",
            "scale_max",
        )
        for field in fields(self):
            if field.name in ("speeds", "drive_table"):
                continue
            value = getattr(self, field.name)
            if not isfinite(value) or value < 0 or (field.name in positive and value == 0):
                raise ValueError(f"Invalid parameter: {field.name}")
        if not (
            self.scale_min <= 1 <= self.scale_max
            and self.min_trust <= 1
            and self.traction_notch <= 15
        ):
            raise ValueError("Invalid scale, trust or notch bounds")
        if bool(self.speeds) != bool(self.drive_table):
            raise ValueError("Provide both speeds and drive_table")
        if self.speeds:
            if (
                len(self.speeds) < 2
                or self.speeds[0] != 0
                or any(not isfinite(v) for v in self.speeds)
                or any(b <= a for a, b in zip(self.speeds, self.speeds[1:]))
                or self.speeds[-1] < self.max_speed
            ):
                raise ValueError("Invalid speed grid")
            if len(self.drive_table) != 31:
                raise ValueError("drive_table requires 31 notch rows")
            for notch, row in zip(range(-15, 16), self.drive_table):
                if len(row) != len(self.speeds) or any(
                    not isfinite(a) or a * notch < 0 or (notch == 0 and a != 0) for a in row
                ):
                    raise ValueError("Invalid drive table value or sign")

    @classmethod
    def from_mapping(cls, mapping):
        """Caller loads YAML/JSON; runtime itself needs no parser dependency."""
        values = dict(mapping)
        values["speeds"] = tuple(values.get("speeds", ()))
        values["drive_table"] = tuple(tuple(row) for row in values.get("drive_table", ()))
        return cls(**values)


@dataclass(frozen=True)
class AdaptiveState:
    drive_scale: float = 1.0

    def __post_init__(self):
        if not isfinite(self.drive_scale) or self.drive_scale <= 0:
            raise ValueError("drive_scale must be positive and finite")


@dataclass(frozen=True)
class DriveEstimate:
    a_model: float
    drive_force_n: float
    net_force_n: float
    resistance_accel: float
    nominal_drive_accel: float
    accel_uncertainty: float
    valid: bool
    saturated: bool


def resistance(v: float, p: Params) -> float:
    return (p.rolling_accel + p.linear_drag * v + p.quadratic_drag * v * v) * min(v / 0.3, 1)


def nominal_drive(u: float, v: float, p: Params) -> float:
    if p.speeds:
        j = min(max(bisect_right(p.speeds, v) - 1, 0), len(p.speeds) - 2)
        fraction = clamp((v - p.speeds[j]) / (p.speeds[j + 1] - p.speeds[j]), 0, 1)

        def at(notch):
            row = p.drive_table[notch + 15]
            return row[j] * (1 - fraction) + row[j + 1] * fraction

        lo = int(u // 1)
        hi = min(lo + 1, 15)
        return at(lo) * (1 - (u - lo)) + at(hi) * (u - lo)
    if u > 0:
        return min(
            p.traction_accel * min(u / p.traction_notch, 1) ** p.traction_exponent,
            p.specific_power / max(v, 0.5),
        )
    return -p.brake_accel * (-u / 15) ** p.brake_exponent


def predict(
    u: float, v: float, dt: float, p: Params, state: AdaptiveState = AdaptiveState()
) -> DriveEstimate:
    """Use previous estimated forward speed, NOT raw wheels. No state mutation.

    u is the latest held notch in [-15, 15]. Invalid input means hold/no model;
    the estimator must honour valid=False and impose its own input timeout.
    """
    if (
        not all(isfinite(x) for x in (u, v, dt))
        or not 0 < dt <= p.max_dt
        or not 0 <= v <= p.max_speed
        or not -15 <= u <= 15
    ):
        return DriveEstimate(0, 0, 0, 0, 0, p.max_accel, False, False)
    nominal = nominal_drive(u, v, p)
    scale = clamp(state.drive_scale, p.scale_min, p.scale_max)
    requested = p.mass_kg * nominal * scale
    force_cap = min(p.max_force_n, p.adhesion_cap * p.mass_kg * 9.81)
    force = clamp(requested, -force_cap, force_cap)
    drag = resistance(v, p)
    raw_accel = force / p.mass_kg - drag
    net_cap = min(p.max_accel, p.max_force_n / p.mass_kg)
    accel = clamp(raw_accel, max(-net_cap, -v / dt), net_cap)
    return DriveEstimate(
        accel,
        force,
        p.mass_kg * accel,
        drag,
        nominal,
        p.accel_uncertainty,
        True,
        force != requested or accel != raw_accel,
    )


def update_params(
    state: AdaptiveState,
    estimate: DriveEstimate,
    observed_accel: float,
    dt: float,
    *,
    trust_wheels: float,
    slip: bool,
    fresh: bool,
    p: Params,
) -> AdaptiveState:
    """Update only from a trusted, time-aligned causal wheel derivative.

    estimate must be the prediction for that SAME measurement interval, before
    wheel correction. State returned here is used on the NEXT tick.
    """
    if (
        not estimate.valid
        or estimate.saturated
        or slip
        or not fresh
        or not all(isfinite(x) for x in (observed_accel, dt, trust_wheels))
        or not 0 < dt <= p.max_dt
        or not p.min_trust <= trust_wheels <= 1
        or abs(observed_accel) > p.max_accel
        or abs(estimate.nominal_drive_accel) < 0.15
    ):
        return state
    phi = estimate.nominal_drive_accel
    residual = clamp(observed_accel - estimate.a_model, -0.5, 0.5)
    step = p.adaptation_rate * dt * phi * residual / (0.1 + phi * phi)
    limit = p.adaptation_max_step_per_s * dt
    scale = clamp(state.drive_scale, p.scale_min, p.scale_max)
    return AdaptiveState(clamp(scale + clamp(step, -limit, limit), p.scale_min, p.scale_max))
