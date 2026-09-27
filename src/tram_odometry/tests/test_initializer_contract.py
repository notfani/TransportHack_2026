"""PositionInitializer is bounded and cannot own speed or wheel calibration."""
import math

import pytest

from tram_estimator import Estimate, Wheel
from tram_position import InitializerConfig, PositionInitializer
from .contract_helpers import checked_map, lla, stamp


def estimate(seconds, *, distance=0., speed=0.):
    return Estimate(stamp(seconds), distance, speed, .02,
                    ((2., .1, .01), (.1, .5, .01), (.01, .01, .1)),
                    1., False, "fixture", ())


def make_initializer(**kwargs):
    localizer = checked_map().as_indexed_localizer()
    return PositionInitializer(localizer, InitializerConfig(route_name="east", **kwargs)), localizer


def test_position_and_velocity_input_leave_complete_estimate_and_wheel_unchanged():
    initializer, localizer = make_initializer()
    wheel = Wheel(stamp(0.), 5.)
    original = estimate(0., speed=5.)
    initializer.receive_fix(stamp(0.), *lla(localizer.calibration, 10.))
    initializer.receive_velocity(stamp(0.), 20., 0., 0.)
    output = initializer.observe(original, raw_front=wheel, raw_rear=wheel)
    assert output.estimate is original
    assert initializer.correct_wheel("front", wheel) is wheel
    assert output.diagnostics["accepted_fix"] == 1
    assert output.diagnostics["rejected_velocity_velocity_disabled"] == 1
    assert output.diagnostics["scale"] == 1.
    assert output.diagnostics["scale_learning_samples"] == 0


def test_pending_dedup_and_history_budgets_remain_bounded():
    initializer, localizer = make_initializer(pending_max=4, dedup_max=8, history_max=6, history_s=1.)
    for i in range(40):
        initializer.receive_fix(stamp(.5+i*.001), *lla(localizer.calibration, 10.))
    output = initializer.observe(estimate(0.))
    assert output.diagnostics["pending_count"] == 4
    assert output.diagnostics["dedup_count"] == 8
    assert output.diagnostics["rejected_fix_queue_overflow"] == 36
    for i in range(1, 31):
        output = initializer.observe(estimate(i*.1))
        assert output.diagnostics["history_count"] <= 6
        assert output.diagnostics["pending_count"] <= 4
        assert output.diagnostics["dedup_count"] <= 8
    assert output.diagnostics["pending_count"] == 0


def test_far_future_and_stale_measurements_cannot_anchor():
    initializer, localizer = make_initializer(history_s=1.)
    initializer.receive_fix(stamp(11.), *lla(localizer.calibration, 10.))
    future = initializer.observe(estimate(0.))
    assert future.anchor is None
    assert future.diagnostics["rejected_fix_future_too_far"] == 1
    for i in range(1, 21):
        initializer.observe(estimate(i*.1))
    initializer.receive_fix(stamp(.1), *lla(localizer.calibration, 10.))
    stale = initializer.observe(estimate(2.1))
    assert stale.anchor is None
    assert stale.diagnostics["rejected_fix_stale"] == 1


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_fix_velocity_and_known_variance_are_rejected_without_state_contamination(bad):
    initializer, localizer = make_initializer()
    initializer.receive_fix(stamp(0.), bad, 0., 0.)
    initializer.receive_velocity(stamp(0.), bad, 0., 0.)
    initializer.receive_fix(stamp(0.), *lla(localizer.calibration, 10.), covariance_type=2, variance_m2=bad)
    original = estimate(0.)
    output = initializer.observe(original)
    assert output.anchor is None and output.estimate is original
    assert output.diagnostics["pending_count"] == 0
    assert output.diagnostics["rejected_fix_invalid"] == 1
    assert output.diagnostics["rejected_fix_invalid_covariance"] == 1
    assert output.diagnostics["rejected_velocity_invalid"] == 1


def test_unknown_covariance_does_not_treat_zero_as_perfect_confidence():
    initializer, localizer = make_initializer()
    initializer.receive_fix(stamp(0.), *lla(localizer.calibration, 10.), covariance_type=0, variance_m2=0.)
    output = initializer.observe(estimate(0.))
    assert output.anchor is not None
    assert initializer.c.position_sigma_m == 3.
    assert output.diagnostics["covariance_calibrated"] is False


def test_duplicate_fix_is_consumed_once_and_same_output_tick_is_refused():
    initializer, localizer = make_initializer()
    payload = lla(localizer.calibration, 10.)
    initializer.receive_fix(stamp(0.), *payload)
    initializer.receive_fix(stamp(0.), *payload)
    original = estimate(0.)
    output = initializer.observe(original)
    assert output.diagnostics["accepted_fix"] == 1
    assert output.diagnostics["rejected_fix_duplicate"] == 1
    with pytest.raises(ValueError, match="new immutable output tick"):
        initializer.observe(original)


@pytest.mark.parametrize("configuration", [
    {"history_s": 11.}, {"pending_max": 0}, {"dedup_max": 1.5},
    {"history_max": True}, {"position_sigma_m": 0.}, {"initial_window_s": math.nan},
])
def test_invalid_resource_and_noise_configuration_is_rejected(configuration):
    with pytest.raises(ValueError):
        InitializerConfig(**configuration)
