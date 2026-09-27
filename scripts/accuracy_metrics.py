"""Offline GNSS reference metrics for a completed ROS bag replay.

The reference mask may inspect neighbouring GNSS fixes. It is used only after
the run and must never be fed back to the odometry node.
"""
from bisect import bisect_left, bisect_right
import math
import statistics


def _summary(values):
    if not values:
        return None
    absolute = sorted(abs(v) for v in values)
    return {
        "n": len(values),
        "mean_m": statistics.mean(values),
        "mae_m": statistics.mean(absolute),
        "rmse_m": math.sqrt(statistics.mean(v * v for v in values)),
        "p95_abs_m": absolute[min(len(absolute) - 1, int(.95 * len(absolute)))],
        "max_abs_m": absolute[-1],
    }


def _valid_fix(fix):
    _, lat, lon, alt, status = fix
    return (status >= 0 and all(math.isfinite(v) for v in (lat, lon, alt))
            and -90 <= lat <= 90 and -180 <= lon <= 180)


def reference_quality_mask(fixes):
    """Match the frozen position-reference edge and one-second padding rule."""
    times = [f[0] for f in fixes]
    quality = [_valid_fix(f) for f in fixes]
    for i in range(len(fixes) - 1):
        a, b = fixes[i], fixes[i + 1]
        dt = (b[0] - a[0]) / 1e9
        bad = not (_valid_fix(a) and _valid_fix(b) and 0 < dt <= .25)
        if not bad:
            lat_a, lat_b = math.radians(a[1]), math.radians(b[1])
            lon_a, lon_b = math.radians(a[2]), math.radians(b[2])
            horizontal = 6_371_000 * math.hypot(
                lat_b - lat_a,
                (lon_b - lon_a) * math.cos((lat_a + lat_b) / 2))
            bad = horizontal / dt > 30 or abs(b[3] - a[3]) / dt > 5
        if bad:
            lo = bisect_left(times, a[0] - 1_000_000_000)
            hi = bisect_right(times, b[0] + 1_000_000_000)
            for j in range(lo, hi):
                quality[j] = False
    return quality


def _nearest(times, target, tolerance_ns=50_000_000):
    i = bisect_left(times, target)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
    if not candidates:
        return None
    j = min(candidates, key=lambda k: abs(times[k] - target))
    return j if abs(times[j] - target) <= tolerance_ns else None


def evaluate_position(positions, fixes, gnss_to_map, *, anchor_ns=None,
                      estimated_distance_m=None):
    """Score (stamp,x,y,yaw,frame) against (stamp,lat,lon,alt,status).

    XY and along/cross are in the route map frame; along/cross use the
    published yaw. The endpoint is scored only when quality GNSS lies within
    one second of the final output. No translation or endpoint fitting occurs.
    """
    positions = sorted(positions, key=lambda p: p[0])
    fixes = sorted(fixes, key=lambda f: f[0])
    times = [p[0] for p in positions]
    quality = reference_quality_mask(fixes)
    records = []
    valid_reference = quality_reference = 0
    for fix, good in zip(fixes, quality):
        if not _valid_fix(fix):
            continue
        valid_reference += 1
        quality_reference += int(good)
        j = _nearest(times, fix[0])
        if j is None or positions[j][4] != "pathgraph":
            continue
        _, x, y, yaw, _ = positions[j]
        rx, ry, _ = gnss_to_map(fix[1], fix[2], fix[3])
        dx, dy = x - rx, y - ry
        along = dx * math.cos(yaw) + dy * math.sin(yaw)
        cross = -dx * math.sin(yaw) + dy * math.cos(yaw)
        records.append((fix[0], math.hypot(dx, dy), along, cross, good))

    def score(rows):
        return {
            "xy_m": _summary([r[1] for r in rows]),
            "along_m": _summary([r[2] for r in rows]),
            "cross_m": _summary([r[3] for r in rows]),
        }

    quality_rows = [r for r in records if r[4]]
    age_bands = ((0, 30), (30, 120), (120, 300), (300, 600),
                 (600, 1200), (1200, 3600), (3600, None))
    by_age = {}
    if anchor_ns is not None:
        for lo, hi in age_bands:
            rows = [r for r in quality_rows
                    if (r[0] - anchor_ns) / 1e9 >= lo
                    and (hi is None or (r[0] - anchor_ns) / 1e9 < hi)]
            by_age[f"{lo}-{hi if hi is not None else 'later'}s"] = score(rows)

    endpoint = None
    if quality_rows and positions:
        last = quality_rows[-1]
        unscored_tail_s = max(0.0, (positions[-1][0] - last[0]) / 1e9)
        endpoint = {
            "last_quality_reference_stamp_ns": last[0],
            "unscored_tail_s": unscored_tail_s,
            "near_final_output": unscored_tail_s <= 1.0,
            "xy_error_m": last[1] if unscored_tail_s <= 1.0 else None,
            "along_error_m": last[2] if unscored_tail_s <= 1.0 else None,
            "cross_error_m": last[3] if unscored_tail_s <= 1.0 else None,
            "error_pct_estimated_distance":
                100 * last[1] / estimated_distance_m
                if unscored_tail_s <= 1.0 and estimated_distance_m is not None
                and estimated_distance_m > 0 else None,
        }

    absolute_outputs = sum(p[4] == "pathgraph" for p in positions)
    return {
        "reference_rule": "valid master fix; adjacent <=0.25s, horizontal <=30m/s, vertical <=5m/s; reject +/-1s around bad edges",
        "alignment": "fixed GNSS-to-map calibration; no endpoint, scale or translation fit",
        "along_cross_basis": "published pose yaw in pathgraph frame",
        "limitations": "GNSS is an imperfect correlated reference; quality mask uses future fixes offline; reported percent uses estimated distance",
        "output_count": len(positions),
        "absolute_output_count": absolute_outputs,
        "absolute_output_share": absolute_outputs / len(positions) if positions else None,
        "valid_reference_count": valid_reference,
        "quality_reference_count": quality_reference,
        "raw_matched_count": len(records),
        "quality_matched_count": len(quality_rows),
        "raw_match_share": len(records) / valid_reference if valid_reference else None,
        "quality_match_share": len(quality_rows) / quality_reference if quality_reference else None,
        "raw": score(records),
        "quality": score(quality_rows),
        "quality_by_time_since_anchor": by_age,
        "endpoint": endpoint,
    }
