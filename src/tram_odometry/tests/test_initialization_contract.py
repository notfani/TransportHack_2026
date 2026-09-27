"""Public initial localization behavior, including late and ambiguous starts."""
from dataclasses import asdict
import math

import pytest

from tram_odometry.bridge import BridgeConfig
from tram_position.heading import LEVER
from .contract_helpers import CLOCK_OFFSET, fix_at, make_bridge, stamp, step


def gnss_config(**kwargs):
    return BridgeConfig(start_mode="gnss", wheel_input_unit="mps", **kwargs)


def test_late_first_master_can_anchor_and_previous_output_stays_immutable():
    bridge, _ = make_bridge(gnss_config(route_name="east"))
    for i in range(101):
        relative = step(bridge, i*.05)
    saved = asdict(relative)
    fix_at(bridge, 5.1, 200.)
    absolute = step(bridge, 5.1)
    assert relative.frame_id == "odom_relative"
    assert not relative.diagnostics["absolute_pose_valid"]
    assert absolute.frame_id == "pathgraph"
    assert absolute.diagnostics["absolute_pose_valid"]
    assert math.isclose(absolute.xyz[0], 200., abs_tol=1e-6)
    assert asdict(relative) == saved
    assert absolute.estimate.speed_mps == 0.


def test_stationary_midroute_master_jitter_does_not_invent_heading():
    bridge, _ = make_bridge(gnss_config(), two_directions=True)
    for i in range(30):
        fix_at(bridge, i*.1, 500.+(i%2)*5.)
        output = step(bridge, i*.1)
    assert bridge.anchor is None
    assert output.frame_id == "odom_relative"
    assert output.estimate.speed_mps == 0.


def test_late_dual_receiver_heading_resolves_stationary_midroute_with_different_clock_epoch():
    bridge, _ = make_bridge(gnss_config(), two_directions=True)
    for i in range(81):
        old = step(bridge, i*.05)
    saved = asdict(old)
    for seconds in (4.1, 4.2):
        fix_at(bridge, seconds, 500.)
        fix_at(bridge, seconds, 500.+LEVER[0], LEVER[1], LEVER[2], source="rover")
        output = step(bridge, seconds)
    assert bridge.anchor.route_name == "east"
    assert output.frame_id == "pathgraph"
    assert output.diagnostics["last_accepted_fix_stamp_ns"] == stamp(4.2)
    assert output.estimate.speed_mps == 0.
    assert asdict(old) == saved and old.frame_id == "odom_relative"


def test_rover_position_alone_cannot_supply_master_anchor():
    bridge, _ = make_bridge(gnss_config(route_name="east"))
    for i in range(20):
        fix_at(bridge, i*.1, 20., source="rover")
        output = step(bridge, i*.1)
    assert bridge.anchor is None
    assert not output.diagnostics["absolute_pose_valid"]
    assert output.estimate.speed_mps == 0.


def test_bad_dual_baseline_cannot_resolve_stationary_ambiguity():
    bridge, _ = make_bridge(gnss_config(), two_directions=True)
    for i in range(10):
        fix_at(bridge, i*.1, 500.)
        fix_at(bridge, i*.1, 550., source="rover")
        output = step(bridge, i*.1)
    assert bridge.anchor is None
    assert output.frame_id == "odom_relative"


def test_initial_refinement_can_change_offset_then_freezes_after_window():
    bridge, _ = make_bridge(gnss_config(route_name="east"))
    for seconds, x in ((0., 10.), (.1, 11.), (.2, 12.)):
        fix_at(bridge, seconds, x)
        output = step(bridge, seconds)
        if seconds == 0.:
            first = output
            original = asdict(first)
    assert math.isclose(output.xyz[0], 11., abs_tol=1e-6)
    assert output.diagnostics["accepted_fix"] == 2
    accepted = output.diagnostics["last_accepted_fix_stamp_ns"]
    fix_at(bridge, 3.1, 25.)
    frozen = step(bridge, 3.1)
    assert math.isclose(frozen.xyz[0], 11., abs_tol=1e-6)
    assert frozen.diagnostics["last_accepted_fix_stamp_ns"] == accepted
    assert frozen.diagnostics["rejected_fix_initialization_complete"] >= 1
    assert asdict(first) == original
    assert frozen.diagnostics["initial_anchor_fix_stamp_ns"] == stamp(0.)


def test_refinement_window_is_relative_to_late_anchor_not_process_start():
    bridge, _ = make_bridge(gnss_config(route_name="east"))
    step(bridge, 0.)
    step(bridge, 5.)
    for seconds, x in ((5.1, 10.), (5.2, 11.), (5.3, 12.)):
        fix_at(bridge, seconds, x)
        output = step(bridge, seconds)
    assert math.isclose(output.xyz[0], 11., abs_tol=1e-6)
    assert output.diagnostics["last_accepted_fix_stamp_ns"] == stamp(5.3)


def test_nondefault_window_changes_refinement_duration_without_accuracy_claim():
    bridges = [make_bridge(gnss_config(route_name="east", initialization_window_s=window))[0]
               for window in (1., 2.)]
    for seconds, x in ((0., 10.), (.1, 10.), (.2, 10.), (1.2, 12.), (1.3, 12.), (1.4, 12.), (1.5, 12.)):
        for bridge in bridges:
            fix_at(bridge, seconds, x)
            step(bridge, seconds)
    assert math.isclose(bridges[0].last_output.xyz[0], 10., abs_tol=1e-6)
    assert math.isclose(bridges[1].last_output.xyz[0], 12., abs_tol=1e-6)
    assert bridges[0].last_output.diagnostics["last_accepted_fix_stamp_ns"] == stamp(.2)
    assert bridges[1].last_output.diagnostics["last_accepted_fix_stamp_ns"] == stamp(1.5)


def test_heading_displacement_parameter_changes_required_motion_without_accuracy_claim():
    bridges = [make_bridge(gnss_config(initial_heading_min_displacement_m=minimum), two_directions=True)[0]
               for minimum in (1., 3.)]
    for i in range(11):
        seconds = i*.1
        for bridge in bridges:
            fix_at(bridge, seconds, 500.+2*seconds)
            step(bridge, seconds, speed=2.)
    assert bridges[0].anchor is not None and bridges[0].anchor.route_name == "east"
    assert bridges[1].anchor is None
    for i in range(11, 21):
        seconds = i*.1
        fix_at(bridges[1], seconds, 500.+2*seconds)
        step(bridges[1], seconds, speed=2.)
    assert bridges[1].anchor is not None and bridges[1].anchor.route_name == "east"


def test_zero_window_explicitly_disables_gnss_initialization():
    bridge, _ = make_bridge(gnss_config(route_name="east", initialization_window_s=0.))
    fix_at(bridge, 0., 10.)
    output = step(bridge, 0.)
    assert output.frame_id == "odom_relative"
    assert output.diagnostics["gnss_initialization_closed"]


def test_same_stamp_duplicate_and_older_refinement_fix_do_not_move_anchor():
    bridge, _ = make_bridge(gnss_config(route_name="east"))
    for seconds, x in ((0., 10.), (.1, 11.), (.2, 12.)):
        fix_at(bridge, seconds, x)
        step(bridge, seconds)
    fix_at(bridge, .2, 99., receipt_seconds=.3)
    fix_at(bridge, .15, 30., receipt_seconds=.3)
    output = step(bridge, .3)
    assert math.isclose(output.xyz[0], 11., abs_tol=1e-6)
    assert output.diagnostics["rejected_fix_duplicate"] == 1
    assert output.diagnostics["rejected_fix_initial_refinement_observed_time_order"] == 1


def test_invalid_fix_does_not_poison_dedup_before_valid_same_stamp():
    bridge, _ = make_bridge(gnss_config(route_name="east"))
    bridge.receive_fix(stamp(0.), math.nan, 0., 0., source="master", clock_ns=stamp(0.)+CLOCK_OFFSET)
    fix_at(bridge, 0., 10.)
    output = step(bridge, 0.)
    assert output.diagnostics["accepted_fix"] == 1
    assert output.diagnostics["rejected_fix_invalid"] == 1
    assert math.isclose(output.xyz[0], 10., abs_tol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"initialization_window_s": -1.}, {"initialization_window_s": 5.1},
    {"initial_heading_min_displacement_m": 0.}, {"initial_heading_min_displacement_m": math.nan},
])
def test_invalid_initialization_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        make_bridge(gnss_config(**kwargs))
