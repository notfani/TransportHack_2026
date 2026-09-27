"""Route constrained position with explicit initialization and coordinate frames.

Runtime depends only on the Python standard library. No GNSS is accepted by
``pose_after``: GNSS is restricted to the initial anchor. The map is fixed.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path


def _ecef(lat_deg: float, lon_deg: float, h: float) -> tuple[float, float, float]:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    a, e2 = 6378137.0, 6.6943799901413165e-3
    n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return ((n+h)*math.cos(lat)*math.cos(lon), (n+h)*math.cos(lat)*math.sin(lon),
            (n*(1-e2)+h)*math.sin(lat))


@dataclass(frozen=True)
class Calibration:
    origin_lla: tuple[float, float, float]
    map_center_xy: tuple[float, float]
    yaw_map_to_enu_rad: float
    enu_at_map_center_xy: tuple[float, float]
    gnss_alt_minus_map_z_m: float

    def gnss_to_map(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
        if not all(math.isfinite(v) for v in (lat, lon, alt)):
            raise ValueError("GNSS must be finite")
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("invalid WGS84 coordinates")
        origin = _ecef(*self.origin_lla)
        point = _ecef(lat, lon, alt)
        dx, dy, dz = (point[i]-origin[i] for i in range(3))
        p, l = (math.radians(v) for v in self.origin_lla[:2])
        east = -math.sin(l)*dx + math.cos(l)*dy
        north = -math.sin(p)*math.cos(l)*dx - math.sin(p)*math.sin(l)*dy + math.cos(p)*dz
        ex, ny = east-self.enu_at_map_center_xy[0], north-self.enu_at_map_center_xy[1]
        c, s = math.cos(self.yaw_map_to_enu_rad), math.sin(self.yaw_map_to_enu_rad)
        return (self.map_center_xy[0]+c*ex+s*ny,
                self.map_center_xy[1]-s*ex+c*ny,
                alt-self.gnss_alt_minus_map_z_m)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw_rad: float
    chainage_m: float
    distance_outside_map_m: float
    region: str
    frame_id: str = "pathgraph"


class DirectedRoute:
    def __init__(self, name: str, points, core_start_m=0.0, core_end_m=None):
        self.name = name
        self.points = tuple(tuple(map(float, p)) for p in points)
        if len(self.points) < 2 or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in self.points):
            raise ValueError("route needs at least two finite XYZ points")
        chainage = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            length = math.dist(a, b)
            if length <= 0:
                raise ValueError("consecutive duplicate map points")
            chainage.append(chainage[-1]+length)
        self.chainage = tuple(chainage)
        self.length_m = chainage[-1]
        self.core_start_m = float(core_start_m)
        self.core_end_m = self.length_m if core_end_m is None else float(core_end_m)

    def pose_at(self, chainage_m: float) -> Pose:
        if not math.isfinite(chainage_m):
            raise ValueError("chainage must be finite")
        bounded = min(max(chainage_m, 0.0), self.length_m)
        i = min(max(bisect_right(self.chainage, bounded)-1, 0), len(self.points)-2)
        alpha = (bounded-self.chainage[i])/(self.chainage[i+1]-self.chainage[i])
        a, b = self.points[i:i+2]
        xyz = tuple(a[k]+alpha*(b[k]-a[k]) for k in range(3))
        region = "original_map" if self.core_start_m <= bounded <= self.core_end_m else "training_extension"
        if chainage_m != bounded:
            region = "outside_map"
        return Pose(*xyz, math.atan2(b[1]-a[1], b[0]-a[0]), chainage_m,
                    abs(chainage_m-bounded), region)

    def project(self, x: float, y: float) -> tuple[float, float]:
        if not all(math.isfinite(v) for v in (x, y)):
            raise ValueError("position must be finite")
        best = (math.inf, 0.0)
        for i, (a, b) in enumerate(zip(self.points, self.points[1:])):
            dx, dy = b[0]-a[0], b[1]-a[1]
            den = dx*dx+dy*dy
            if den == 0:
                continue
            f = min(max(((x-a[0])*dx+(y-a[1])*dy)/den, 0.0), 1.0)
            distance2 = (x-a[0]-f*dx)**2+(y-a[1]-f*dy)**2
            if distance2 < best[0]:
                best = distance2, self.chainage[i]+f*(self.chainage[i+1]-self.chainage[i])
        return best[1], math.sqrt(best[0])

    def projection_candidates(self, x: float, y: float, max_lateral_m: float,
                              heading_rad: float | None = None) -> list[tuple[float, float, float]]:
        """Project onto each connected branch within the initialization radius.

        Return (chainage, squared lateral distance, heading mismatch) per branch.
        Squared distances preserve the exact nearest-segment tie ordering.
        Choosing the nearest segment *before* checking heading loses valid
        branches on loops. With a heading, retain only compatible segments.
        Adjacent compatible segments form one branch;
        separate runs remain separate hypotheses even in the same route.
        This scan runs only during initial anchoring, never in pose_after.
        """
        candidates, best = [], None
        for i, (a, b) in enumerate(zip(self.points, self.points[1:])):
            # Most map segments cannot intersect the initial uncertainty disk.
            # Reject their bounding boxes before projection/trigonometry.
            if ((x < a[0]-max_lateral_m and x < b[0]-max_lateral_m)
                    or (x > a[0]+max_lateral_m and x > b[0]+max_lateral_m)
                    or (y < a[1]-max_lateral_m and y < b[1]-max_lateral_m)
                    or (y > a[1]+max_lateral_m and y > b[1]+max_lateral_m)):
                if best is not None:
                    candidates.append(best)
                    best = None
                continue
            dx, dy = b[0]-a[0], b[1]-a[1]
            den = dx*dx+dy*dy
            mismatch = 0.0
            if heading_rad is not None:
                delta_heading = heading_rad-math.atan2(dy, dx)
                mismatch = abs(math.atan2(math.sin(delta_heading), math.cos(delta_heading)))
            f = min(max(((x-a[0])*dx+(y-a[1])*dy)/den, 0.0), 1.0) if den else 0.0
            distance2 = (x-a[0]-f*dx)**2+(y-a[1]-f*dy)**2
            if den == 0 or mismatch > math.pi/3 or math.sqrt(distance2) > max_lateral_m:
                if best is not None:
                    candidates.append(best)
                    best = None
                continue
            candidate = (self.chainage[i]+f*(self.chainage[i+1]-self.chainage[i]),
                         distance2, mismatch)
            if best is None or distance2 < best[1]:
                best = candidate
        if best is not None:
            candidates.append(best)
        return candidates


@dataclass(frozen=True)
class Anchor:
    route_name: str
    chainage_m: float
    lateral_error_m: float
    selection_reason: str


class RouteLocalizer:
    def __init__(self, calibration: Calibration, routes: tuple[DirectedRoute, ...]):
        self.calibration, self.routes = calibration, routes

    @classmethod
    def load(cls, filename: str | Path):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        calibration = Calibration(**data["calibration"])
        return cls(calibration, tuple(DirectedRoute(**r) for r in data["routes"]))

    def anchor(self, lat: float, lon: float, alt: float, *, route_name: str | None = None,
               heading_enu_rad: float | None = None, max_lateral_m: float = 30.0) -> Anchor:
        """Initial fix only; explicit direction, moving heading, or terminal prior.

        A stationary mid-route start without direction is deliberately rejected.
        Terminal prior covers the learned prefix plus the first 100 m of core.
        """
        if heading_enu_rad is not None and not math.isfinite(heading_enu_rad):
            raise ValueError("heading must be finite")
        if not math.isfinite(max_lateral_m) or max_lateral_m < 0:
            raise ValueError("max_lateral_m must be finite and nonnegative")
        x, y, _ = self.calibration.gnss_to_map(lat, lon, alt)
        candidates = []
        for route in self.routes:
            if route_name is not None and route.name != route_name:
                continue
            if heading_enu_rad is not None:
                heading_map = heading_enu_rad-self.calibration.yaw_map_to_enu_rad
                for s, distance2, mismatch in route.projection_candidates(x, y, max_lateral_m, heading_map):
                    lateral = math.sqrt(distance2)
                    candidates.append((lateral+10*mismatch,
                                       Anchor(route.name, s, lateral, "initial_heading")))
                continue
            nearby = route.projection_candidates(x, y, max_lateral_m)
            if not nearby:
                continue
            # Without heading retain the original global-nearest projection,
            # including its terminal-prior rejection. Do not substitute a
            # farther departing branch for the actual nearest branch.
            s, distance2, _ = min(nearby, key=lambda item: item[1])
            lateral = math.sqrt(distance2)
            if route_name is not None:
                score, reason = lateral, "explicit_direction"
            elif s <= route.core_start_m+100:
                score, reason = lateral, "terminal_prior"
            else:
                continue
            candidates.append((score, Anchor(route.name, s, lateral, reason)))
        if not candidates:
            raise ValueError("initial route/direction is unavailable or outside map")
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0]-candidates[0][0] < 1.0:
            raise ValueError("initial route is ambiguous")
        return candidates[0][1]

    def pose_after(self, anchor: Anchor, distance_since_anchor_m: float) -> Pose:
        """Pure map interpolation. No GNSS update or test-specific transform."""
        route = next((r for r in self.routes if r.name == anchor.route_name), None)
        if route is None:
            raise ValueError("unknown anchor route")
        return route.pose_at(anchor.chainage_m+distance_since_anchor_m)
