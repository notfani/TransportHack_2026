"""Validated local map product and bounded explicit-topology research adapter.

No network, extrapolation, state-estimator corrections, or inferred junctions.
Distances are signed displacement; unwrapped travel must remain in the estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .route_localizer import Calibration, DirectedRoute, Pose, RouteLocalizer

MAX_MAP_BYTES = 16_000_000
MAX_POINTS = 100_000
MAX_ROUTES = 128
MAX_HYPOTHESES = 8
MAX_TRANSITIONS = 64
JUNCTION_TOLERANCE_M = 0.25


def _finite(values, size, label):
    if not isinstance(values, (tuple, list)) or len(values) != size:
        raise ValueError(f"{label}: expected {size} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f"{label}: values must be finite numbers")


@dataclass(frozen=True)
class MapPose:
    valid: bool
    reason: str
    pose: Pose | None
    route_name: str
    chainage_m: float
    outside_distance_m: float






class _IndexedDirectedRoute(DirectedRoute):
    """API-compatible shallow facade; geometry tuples remain shared."""
    def __init__(self, checked, route):
        self.__dict__.update(route.__dict__)
        self._checked = checked
        self._route_name = route.name

    def projection_candidates(self, x, y, max_lateral_m, heading_rad=None):
        # Preserve original endpoint projection semantics for existing callers.
        # The supervisor still applies its explicit endpoint/outside gate.
        return self._checked._indexed_candidates(self._route_name, x, y, max_lateral_m,
                                                 heading_rad, reject_endpoint_tangent=False)


class CheckedRouteMap:
    """Map contract wrapper. Schema1 has documented legacy metre/radian units.

    Schema2 additionally requires units, reference_point, height_reference and
    provenance. All schemas retain the same calibration and directed routes.
    Schema2 optional transitions contain [source_route, destination_route].
    A transition is explicit permission to traverse an actual connected path;
    endpoint proximity by itself is never interpreted as connectivity.
    """
    def __init__(self, localizer: RouteLocalizer, *, frame_id="pathgraph", transitions=(), metadata=None):
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        self.localizer = localizer
        self.calibration = localizer.calibration
        self.routes = localizer.routes
        self.frame_id = frame_id
        self.metadata = metadata or {}
        self.by_name = {r.name: r for r in self.routes}
        if len(self.by_name) != len(self.routes) or not self.routes or len(self.routes) > MAX_ROUTES:
            raise ValueError("route ids must be unique; map needs 1..128 routes")
        total_points = 0
        for route in self.routes:
            if not isinstance(route.name, str) or not route.name:
                raise ValueError("route id must be a nonempty string")
            total_points += len(route.points)
            if not (math.isfinite(route.length_m) and math.isfinite(route.core_start_m) and math.isfinite(route.core_end_m)
                    and 0 <= route.core_start_m <= route.core_end_m <= route.length_m + 1e-6):
                raise ValueError("invalid core interval")
            for i, (a, b) in enumerate(zip(route.points, route.points[1:])):
                dx, dy = b[0]-a[0], b[1]-a[1]
                den = dx*dx+dy*dy
                if not math.isfinite(den) or den <= 0:
                    raise ValueError("segment horizontal norm must be positive and representable")
                if route.chainage[i+1] <= route.chainage[i]:
                    raise ValueError("segment chainage increment is not representable")
        if total_points > MAX_POINTS:
            raise ValueError("map point budget exceeded")
        c = self.calibration
        _finite(c.origin_lla, 3, "origin_lla")
        _finite(c.map_center_xy, 2, "map_center_xy")
        _finite(c.enu_at_map_center_xy, 2, "enu_at_map_center_xy")
        _finite((c.yaw_map_to_enu_rad, c.gnss_alt_minus_map_z_m), 2, "calibration")
        if not (-90 < c.origin_lla[0] < 90 and -180 <= c.origin_lla[1] <= 180):
            raise ValueError("invalid georeference origin")
        # Coarse consecutive segment blocks preserve candidate order and runs,
        # while avoiding a Python scan over every point during GNSS return.
        self.blocks = {}
        for route in self.routes:
            blocks = []
            for start in range(0, len(route.points)-1, 64):
                stop = min(start+64, len(route.points)-1)
                points = route.points[start:stop+1]
                blocks.append((start, stop, min(p[0] for p in points), max(p[0] for p in points),
                               min(p[1] for p in points), max(p[1] for p in points)))
            self.blocks[route.name] = tuple(blocks)
        self.forward = {name: [] for name in self.by_name}
        self.backward = {name: [] for name in self.by_name}
        if len(transitions) > 256:
            raise ValueError("transition budget exceeded")
        seen = set()
        for edge in transitions:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError("transition must contain source and destination")
            a, b = edge
            if not isinstance(a, str) or not isinstance(b, str) or a not in self.by_name or b not in self.by_name or (a, b) in seen:
                raise ValueError("unknown or duplicate transition")
            seen.add((a, b))
            if math.dist(self.by_name[a].points[-1], self.by_name[b].points[0]) > JUNCTION_TOLERANCE_M:
                raise ValueError("transition crosses unmapped gap")
            self.forward[a].append(b)
            self.backward[b].append(a)

    @classmethod
    def load(cls, path, *, expected_sha256=None, allowed_training_bags=None):
        path = Path(path)
        if path.stat().st_size > MAX_MAP_BYTES:
            raise ValueError("map byte budget exceeded")
        raw = path.read_bytes()
        if len(raw) > MAX_MAP_BYTES:
            raise ValueError("map byte budget exceeded")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("map SHA256 mismatch")
        data = json.loads(raw)
        return cls.from_data(data, allowed_training_bags=allowed_training_bags)

    @classmethod
    def from_data(cls, data: dict[str, Any], *, allowed_training_bags=None):
        if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] not in (1, 2):
            raise ValueError("unsupported map schema")
        schema = data["schema"]
        if "units" in data and data["units"] != {"length": "m", "angle": "rad"}:
            raise ValueError("map requires metres/radians")
        if schema == 2:
            if data.get("units") != {"length": "m", "angle": "rad"}:
                raise ValueError("map requires metres/radians")
            for key in ("reference_point", "height_reference"):
                if not isinstance(data.get(key), str) or not data[key].strip():
                    raise ValueError(f"missing {key}")
            if not isinstance(data.get("provenance"), dict) or not data["provenance"].get("source"):
                raise ValueError("missing provenance source")
        elif not isinstance(data.get("training_bags"), list) or not data["training_bags"]:
            raise ValueError("legacy map lacks training provenance")
        training = data.get("training_bags", [])
        if any(not isinstance(b, str) or not b for b in training):
            raise ValueError("invalid training bag id")
        if allowed_training_bags is not None and not set(training).issubset(allowed_training_bags):
            raise ValueError("map training sources cross allowed split")
        routes = data.get("routes")
        if not isinstance(routes, list) or not 1 <= len(routes) <= MAX_ROUTES:
            raise ValueError("map route budget exceeded")
        if any(not isinstance(r, dict) or not isinstance(r.get("points"), list) for r in routes):
            raise ValueError("invalid route structure")
        if sum(len(r["points"]) for r in routes) > MAX_POINTS:
            raise ValueError("map point budget exceeded")
        for route in routes:
            for point in route["points"]:
                # DirectedRoute performs shape/finiteness validation once below.
                # Reject JSON strings/bools without repeating its finite scan.
                if not isinstance(point, (list, tuple)) or any(type(v) not in (int, float) for v in point):
                    raise ValueError("map points must be numeric XYZ")
        if not isinstance(data.get("transitions", []), (tuple, list)):
            raise ValueError("invalid transitions structure")
        try:
            calibration = Calibration(**data["calibration"])
            localizer = RouteLocalizer(calibration, tuple(DirectedRoute(**r) for r in routes))
        except (TypeError, KeyError, OverflowError) as exc:
            raise ValueError("invalid map structure") from exc
        return cls(localizer, frame_id=data.get("frame_id", ""),
                   transitions=data.get("transitions", []), metadata={k: v for k, v in data.items() if k != "routes"})

    def pose(self, route_name: str, chainage_m: float) -> MapPose:
        if route_name not in self.by_name:
            return MapPose(False, "unknown_route", None, route_name, chainage_m, math.nan)
        if not math.isfinite(chainage_m):
            return MapPose(False, "invalid_chainage", None, route_name, chainage_m, math.nan)
        route = self.by_name[route_name]
        outside = max(-chainage_m, chainage_m-route.length_m, 0.0)
        if outside > 0:
            return MapPose(False, "outside_map", None, route_name, chainage_m, outside)
        return MapPose(True, "ok", replace(route.pose_at(chainage_m), frame_id=self.frame_id),
                       route_name, chainage_m, 0.0)

    def as_indexed_localizer(self):
        """Return the legacy localizer API with indexed route candidates.

        Route geometry, pose_at/project and anchor semantics stay unchanged,
        including diagnostic endpoint projections. Use this adapter to avoid
        bypassing the index in existing route.projection_candidates callers.
        Published pose validity remains the supervisor's responsibility; use
        CheckedRouteMap.pose for the explicit map-product no-clamp contract.
        """
        return RouteLocalizer(self.calibration, tuple(_IndexedDirectedRoute(self, r) for r in self.routes))

    def projection_candidates(self, route_name, x, y, max_lateral_m, heading_rad=None):
        return self._indexed_candidates(route_name, x, y, max_lateral_m, heading_rad,
                                        reject_endpoint_tangent=True)

    def _indexed_candidates(self, route_name, x, y, max_lateral_m, heading_rad=None,
                            *, reject_endpoint_tangent=True):
        """Exclude endpoint-clamped projections beyond endpoint tangent planes.

        Does not certify GNSS integrity or map accuracy. Candidate ambiguity is
        preserved; caller must compare the *entire* returned candidate set.
        """
        if not all(math.isfinite(v) for v in (x, y, max_lateral_m)) or max_lateral_m < 0:
            raise ValueError("projection inputs must be finite; radius nonnegative")
        if heading_rad is not None and not math.isfinite(heading_rad):
            raise ValueError("heading must be finite")
        route = self.by_name[route_name]
        candidates, best = [], None
        for start, stop, xmin, xmax, ymin, ymax in self.blocks[route_name]:
            if x < xmin-max_lateral_m or x > xmax+max_lateral_m or y < ymin-max_lateral_m or y > ymax+max_lateral_m:
                if best is not None:
                    candidates.append(best)
                    best = None
                continue
            for i in range(start, stop):
                a, b = route.points[i], route.points[i+1]
                if ((x < a[0]-max_lateral_m and x < b[0]-max_lateral_m)
                        or (x > a[0]+max_lateral_m and x > b[0]+max_lateral_m)
                        or (y < a[1]-max_lateral_m and y < b[1]-max_lateral_m)
                        or (y > a[1]+max_lateral_m and y > b[1]+max_lateral_m)):
                    if best is not None:candidates.append(best);best=None
                    continue
                dx, dy = b[0]-a[0], b[1]-a[1]
                den = dx*dx+dy*dy
                f = min(max(((x-a[0])*dx+(y-a[1])*dy)/den,0.),1.)
                distance2 = (x-a[0]-f*dx)**2+(y-a[1]-f*dy)**2
                mismatch = 0.
                if heading_rad is not None:
                    delta = heading_rad-math.atan2(dy,dx)
                    mismatch = abs(math.atan2(math.sin(delta),math.cos(delta)))
                if math.sqrt(distance2) > max_lateral_m or mismatch > math.pi/3:
                    if best is not None:
                        candidates.append(best)
                        best = None
                    continue
                item = (route.chainage[i]+f*(route.chainage[i+1]-route.chainage[i]),distance2,mismatch)
                if best is None or distance2 < best[1]:
                    best = item
        if best is not None:
            candidates.append(best)
        if not reject_endpoint_tangent:return candidates
        result = []
        for candidate in candidates:
            s = candidate[0]
            if s <= 1e-9:
                a, b = route.points[:2]
                if (x-a[0])*(b[0]-a[0]) + (y-a[1])*(b[1]-a[1]) < -1e-9:
                    continue
            elif s >= route.length_m-1e-9:
                a, b = route.points[-2:]
                if (x-b[0])*(b[0]-a[0]) + (y-b[1])*(b[1]-a[1]) > 1e-9:
                    continue
            result.append(candidate)
        return result

