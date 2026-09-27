"""Portable causality, units, manual frame and watchdog contracts."""
import math

from tram_odometry.bridge import BridgeConfig
from .contract_helpers import BASE, CLOCK_OFFSET, make_bridge, stamp, step


def test_previous_command_owns_elapsed_interval_and_kmh_is_converted_once():
    bridge, drive = make_bridge(BridgeConfig(start_mode="relative", wheel_input_unit="kmh"))
    first = step(bridge, 0., speed=18., command=10)
    assert first.estimate.speed_mps == 5.
    step(bridge, .05, speed=18., command=-10)
    assert drive.calls[-1][0] == 10
    step(bridge, .1, speed=18., command=0)
    assert drive.calls[-1][0] == -10


def test_explicit_mps_units_preserve_five_metres_per_second():
    bridge, _ = make_bridge()
    assert step(bridge, 0., speed=5.).estimate.speed_mps == 5.


def test_watchdog_uses_elapsed_arrival_clock_and_paused_clock_cannot_advance():
    bridge, _ = make_bridge()
    step(bridge, 0., speed=5.)
    outputs = [bridge.tick(BASE+CLOCK_OFFSET+i*20_000_000) for i in range(1, 31)]
    outputs = [output for output in outputs if output is not None]
    assert outputs[0].estimate.stamp_ns == BASE+60_000_000
    assert max(b.estimate.stamp_ns-a.estimate.stamp_ns for a, b in zip(outputs, outputs[1:])) <= 100_000_000
    assert outputs[-1].diagnostics["command_stale"]
    assert bridge.tick(BASE+CLOCK_OFFSET+600_000_000) is None


def test_older_commands_and_wheels_do_not_rewind_state():
    bridge, _ = make_bridge()
    step(bridge, 0., speed=5., command=1)
    current = step(bridge, .05, speed=5., command=2)
    assert bridge.receive_command(stamp(.01), -15, stamp(.06)+CLOCK_OFFSET) is None
    bridge.receive_wheel("front", stamp(.03), 99., stamp(.07)+CLOCK_OFFSET)
    assert bridge.command == 2
    assert bridge.front.speed_mps == 5.
    assert bridge.last_output is current


def test_invalid_wheel_and_command_do_not_replace_valid_input():
    bridge, _ = make_bridge()
    original = step(bridge, 0., speed=5., command=2)
    bridge.receive_wheel("front", stamp(.01), math.nan, stamp(.01)+CLOCK_OFFSET)
    assert bridge.receive_command(stamp(.01), math.nan, stamp(.01)+CLOCK_OFFSET) is None
    assert bridge.front.speed_mps == 5. and bridge.command == 2
    assert bridge.last_output is original
    output = step(bridge, .05, speed=5., command=2)
    assert output.diagnostics["invalid_input"] == 2
    assert all(math.isfinite(value) for value in (output.estimate.speed_mps, output.estimate.distance_m, *output.xyz))


def test_stream_of_late_commands_cannot_suppress_watchdog():
    bridge, _ = make_bridge()
    step(bridge, 0.)
    assert bridge.tick(stamp(1.)+CLOCK_OFFSET) is not None
    outputs = []
    for i in range(1, 11):
        clock = stamp(1.+i*.05)+CLOCK_OFFSET
        assert bridge.receive_command(stamp(i*.05), 1, clock) is None
        outputs.append(bridge.tick(clock))
    assert all(output is not None for output in outputs)
    assert outputs[-1].estimate.stamp_ns == stamp(1.5)


def test_manual_chainage_and_output_transform_are_applied_once():
    config = BridgeConfig(start_mode="route_chainage", route_name="east", start_chainage_m=10.,
                          map_frame_id="configured_map", output_yaw_rad=math.pi/2,
                          output_x_m=20., output_z_m=4.)
    bridge, _ = make_bridge(config)
    output = step(bridge, 0.)
    assert math.isclose(output.xyz[0], 20.)
    assert math.isclose(output.xyz[1], 10.)
    assert output.xyz[2] == 4.
    assert math.isclose(output.yaw_rad, math.pi/2)
    assert output.frame_id == "configured_map"
    assert output.pose_covariance[0] >= 9.
    assert output.diagnostics["covariance_calibrated"] is False


def test_manual_map_pose_selects_explicit_route_and_relative_pose_keeps_its_origin():
    bridge, _ = make_bridge(BridgeConfig(start_mode="map_pose", route_name="east", start_x_m=15., start_y_m=1.))
    output = step(bridge, 0.)
    assert output.xyz == (15., 0., 0.)
    assert output.diagnostics["initialization"] == "configured_map_pose"
    relative, _ = make_bridge(BridgeConfig(start_mode="relative", wheel_input_unit="mps",
                                          start_x_m=20., start_y_m=30., start_z_m=4., start_yaw_rad=math.pi/2))
    step(relative, 0., speed=5.)
    output = step(relative, .1, speed=5.)
    assert output.frame_id == "odom_relative"
    assert math.isclose(output.xyz[0], 20.)
    assert math.isclose(output.xyz[1], 30.+output.estimate.distance_m)
    assert output.xyz[2] == 4.


def test_role2_command_expires_after_point_two_seconds():
    bridge, drive = make_bridge(BridgeConfig(start_mode="relative"))
    step(bridge, 0., speed=5., command=5)
    out = bridge.tick(stamp(.22)+CLOCK_OFFSET)
    assert out is not None
    assert out.diagnostics["command_stale"] is True
    assert out.diagnostics["command_used"] == 0
    assert not drive.calls
