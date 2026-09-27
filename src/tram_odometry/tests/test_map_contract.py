"""Finite coverage, schema/provenance and indexed branch-preservation contracts."""
from dataclasses import asdict
import hashlib
import json
import math

import pytest

from tram_odometry.bridge import BridgeConfig
from tram_position import CheckedRouteMap
from tram_position.route_localizer import DirectedRoute, RouteLocalizer
from .contract_helpers import checked_map, make_bridge, step


def map_document():
    checked = checked_map()
    return dict(schema=2, frame_id="test_map", units={"length": "m", "angle": "rad"},
                reference_point="gnss_master_antenna", height_reference="synthetic_metres",
                provenance={"source": "synthetic_contract_fixture"}, training_bags=["train_fixture"],
                calibration=asdict(checked.calibration),
                routes=[dict(name="east", points=[[0., 0., 0.], [1000., 0., 0.]])])


def test_outside_map_returns_no_absolute_pose_instead_of_clamped_endpoint():
    checked = checked_map(length=100.)
    inside = checked.pose("east", 100.)
    outside = checked.pose("east", 101.)
    assert inside.valid and inside.pose.x == 100.
    assert not outside.valid and outside.pose is None
    assert outside.reason == "outside_map" and outside.outside_distance_m == 1.
    assert checked.pose("east", math.nan).reason == "invalid_chainage"
    assert checked.pose("unknown", 10.).reason == "unknown_route"


def test_bridge_map_exit_switches_named_frame_and_keeps_unbounded_distance():
    bridge, _ = make_bridge(BridgeConfig(start_mode="route_chainage", route_name="east",
                                         start_chainage_m=95., wheel_input_unit="mps"), length=100.)
    first = step(bridge, 0., speed=5.)
    for i in range(1, 61):
        output = step(bridge, i*.05, speed=5.)
    assert first.frame_id == "pathgraph" and first.diagnostics["absolute_pose_valid"]
    assert output.frame_id == "odom_relative" and not output.diagnostics["absolute_pose_valid"]
    assert output.diagnostics["route_region"] == "outside_map_relative"
    assert output.diagnostics["chainage_m"] > 105.
    assert output.diagnostics["outside_map_m"] > 5.
    assert output.estimate.distance_m > 10.
    assert math.isclose(output.xyz[0], output.estimate.distance_m)
    assert all(math.isfinite(value) for value in output.pose_covariance)


def test_checked_projection_rejects_beyond_endpoint_while_legacy_initial_api_is_explicit():
    checked = checked_map(length=100.)
    assert checked.projection_candidates("east", -1., 0., 10.) == []
    assert checked.projection_candidates("east", 101., 0., 10.) == []
    # Initial GNSS retains the documented noisy endpoint projection assumption.
    # Strict coverage applies to the published pose after propagation.
    indexed = checked.as_indexed_localizer()
    assert indexed.routes[0].projection_candidates(-1., 0., 10.)[0][0] == 0.


def test_index_keeps_disjoint_parallel_branch_candidates_in_original_order():
    vertices = [(0., 0., 0.), (100., 0., 0.), (100., 100., 0.), (0., 100., 0.), (0., 2., 0.), (100., 2., 0.)]
    points = []
    for a, b in zip(vertices, vertices[1:]):
        for i in range(32):
            points.append(tuple(a[k]+(b[k]-a[k])*i/32 for k in range(3)))
    points.append(vertices[-1])
    route = DirectedRoute("loop", points)
    checked = CheckedRouteMap(RouteLocalizer(checked_map().calibration, (route,)))
    expected = route.projection_candidates(50., 1., 3., 0.)
    indexed = checked.as_indexed_localizer().routes[0].projection_candidates(50., 1., 3., 0.)
    assert len(expected) == 2 and expected[1][0]-expected[0][0] > 300.
    assert indexed == expected


def test_schema_provenance_units_and_file_digest_are_validated(tmp_path):
    document = map_document()
    checked = CheckedRouteMap.from_data(document, allowed_training_bags={"train_fixture"})
    assert checked.pose("east", 10.).pose.frame_id == "test_map"
    path = tmp_path/"map.json"
    raw = json.dumps(document).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert CheckedRouteMap.load(path, expected_sha256=digest).pose("east", 10.).valid
    with pytest.raises(ValueError, match="SHA256"):
        CheckedRouteMap.load(path, expected_sha256="0"*64)
    with pytest.raises(ValueError, match="allowed split"):
        CheckedRouteMap.from_data(document, allowed_training_bags={"different_training_source"})


@pytest.mark.parametrize("change", [
    {"schema": 3}, {"schema": True}, {"units": {"length": "ft", "angle": "rad"}},
    {"reference_point": ""}, {"height_reference": ""}, {"provenance": {}},
    {"frame_id": ""}, {"training_bags": [False]},
])
def test_malformed_map_metadata_is_rejected(change):
    document = map_document()
    document.update(change)
    with pytest.raises(ValueError):
        CheckedRouteMap.from_data(document)


@pytest.mark.parametrize("point", [[True, 0., 0.], ["0", 0., 0.], [math.nan, 0., 0.], [math.inf, 0., 0.]])
def test_non_numeric_or_nonfinite_map_geometry_is_rejected(point):
    document = map_document()
    document["routes"][0]["points"][0] = point
    with pytest.raises(ValueError):
        CheckedRouteMap.from_data(document)


def test_map_cannot_infer_a_transition_across_an_unmapped_gap():
    document = map_document()
    document["routes"].append(dict(name="next", points=[[1002., 0., 0.], [1100., 0., 0.]]))
    document["transitions"] = [["east", "next"]]
    with pytest.raises(ValueError, match="unmapped gap"):
        CheckedRouteMap.from_data(document)
