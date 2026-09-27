#!/usr/bin/env python3
"""Compare installed model adapters on one bag without ROS or network access.

This is a receive-order mathematical replay. Its ideal 20 ms watchdog clock
does not measure DDS delivery, wall-clock frequency, or ROS callback latency.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "tram_odometry"))

from accuracy_metrics import evaluate_position
from tram_estimator.drive import TableDrive
from tram_odometry.bridge import Bridge
from tram_odometry.hybrid_drive import HybridDrive
from tram_position import CheckedRouteMap

TOPICS = {
    "/vehicle/front_bogie_velocity": "front",
    "/vehicle/rear_bogie_velocity": "rear",
    "/vehicle/driver_position_cmd": "command",
    "/sensing/gnss/master/fix": "master_fix",
    "/sensing/gnss/rover/fix": "rover_fix",
    "/sensing/gnss/master/vel": "master_velocity",
}


class CDR:
    def __init__(self, data):
        if data[:4] != b"\x00\x01\x00\x00":
            raise ValueError("Expected little-endian CDR")
        self.data, self.offset = data, 4

    def read(self, fmt, alignment):
        self.offset = 4 + ((self.offset - 4 + alignment - 1) // alignment) * alignment
        value = struct.unpack_from("<" + fmt, self.data, self.offset)[0]
        self.offset += struct.calcsize(fmt)
        return value

    def header_stamp(self):
        stamp = self.read("i", 4) * 1_000_000_000 + self.read("I", 4)
        length = self.read("I", 4)
        self.offset += length
        return stamp


def read_events(bag):
    events = []
    for db in sorted(bag.glob("*.db3")):
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as con:
            names = dict(con.execute("SELECT id,name FROM topics"))
            for received, rowid, topic_id, raw in con.execute(
                    "SELECT timestamp,id,topic_id,data FROM messages ORDER BY timestamp,id"):
                kind = TOPICS.get(names[topic_id])
                if kind is None:
                    continue
                cdr = CDR(raw)
                stamp = cdr.header_stamp()
                if kind in ("front", "rear"):
                    payload = cdr.read("d", 8)
                elif kind == "command":
                    payload = cdr.read("b", 1)
                elif kind.endswith("fix"):
                    status = cdr.read("b", 1)
                    cdr.read("H", 2)
                    lat, lon, alt = (cdr.read("d", 8) for _ in range(3))
                    covariance = [cdr.read("d", 8) for _ in range(9)]
                    covariance_type = cdr.read("B", 1)
                    payload = (lat, lon, alt, status, covariance_type,
                               max(abs(v) for v in covariance))
                else:
                    payload = tuple(cdr.read("d", 8) for _ in range(6))
                events.append((received, rowid, kind, stamp, payload))
    events.sort(key=lambda row: (row[0], row[1]))
    return events


def speed_score(outputs, reference):
    outputs.sort()
    stamps = [row[0] for row in outputs]
    errors = []
    for stamp, speed in reference:
        i = bisect_left(stamps, stamp)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(stamps)]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(stamps[k] - stamp))
        if abs(stamps[j] - stamp) <= 50_000_000:
            errors.append(outputs[j][1] - speed)
    return ({"n": len(errors), "rmse_mps": math.sqrt(sum(e*e for e in errors)/len(errors)),
             "mae_mps": sum(abs(e) for e in errors)/len(errors),
             "bias_mps": sum(errors)/len(errors)} if errors else None)


def replay(events, model_kind, localizer):
    config = ROOT / "src" / "tram_odometry" / "config"
    drive = (HybridDrive(config / "hybrid.json") if model_kind == "hybrid" else
             TableDrive(config / "drive_table.json"))
    bridge = Bridge(drive, localizer)
    positions, velocities, fixes, reference_speeds = [], [], [], []
    model_counts = Counter()

    def collect(out):
        if out is None:
            return
        e = out.estimate
        positions.append((e.stamp_ns, out.xyz[0], out.xyz[1], out.yaw_rad, out.frame_id))
        velocities.append((e.stamp_ns, e.speed_mps))
        model_counts[out.diagnostics["drive_model_source"]] += 1

    if not events:
        raise ValueError("No supported messages in bag")
    tick = events[0][0]
    for received, _, kind, stamp, payload in events:
        while tick <= received:
            collect(bridge.tick(tick))
            tick += 20_000_000
        if kind in ("front", "rear"):
            bridge.receive_wheel(kind, stamp, payload, received)
        elif kind == "command":
            collect(bridge.receive_command(stamp, payload, received))
        elif kind.endswith("fix"):
            lat, lon, alt, status, covariance_type, variance = payload
            bridge.receive_fix(stamp, lat, lon, alt, status,
                               source="master" if kind == "master_fix" else "rover",
                               covariance_type=covariance_type, variance_m2=variance,
                               clock_ns=received)
            if kind == "master_fix":
                fixes.append((stamp, lat, lon, alt, status))
        elif kind == "master_velocity":
            vx, vy, vz = payload[:3]
            bridge.receive_velocity(stamp, vx, vy, vz, clock_ns=received)
            reference_speeds.append((stamp, math.sqrt(vx*vx + vy*vy + vz*vz)))
    until = events[-1][0] + 500_000_000
    while tick <= until:
        collect(bridge.tick(tick))
        tick += 20_000_000

    last = bridge.fusion.last_accepted_fix_stamp_ns or None
    distance = (abs(bridge.estimator.x[0] - bridge.fusion.odom_distance_at_correction)
                if last is not None else None)
    return {
        "model": model_kind,
        "output_count": len(positions),
        "drive_model_tick_counts": dict(model_counts),
        "hybrid_drive_scale_final": (drive.state.adaptive.drive_scale
                                     if model_kind == "hybrid" else None),
        "hybrid_memory_samples_final": (len(drive.history)
                                        if model_kind == "hybrid" else None),
        "speed_raw": speed_score(velocities, reference_speeds),
        "position": evaluate_position(positions, fixes, localizer.calibration.gnss_to_map,
                                      anchor_ns=last, estimated_distance_m=distance),
        "last_accepted_fix_stamp_ns": last,
        "distance_since_anchor_m": distance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    events = read_events(args.bag)
    map_path = ROOT / "src" / "tram_odometry" / "config" / "route_bundle.json"
    localizer = CheckedRouteMap.load(map_path).as_indexed_localizer()
    report = {
        "bag": str(args.bag.resolve()),
        "mode": "receive-order offline Bridge; ideal 20ms watchdog",
        "limitations": "No ROS scheduling, DDS, GNSS quality certificate or hidden-test validation",
        "models": [replay(events, kind, localizer) for kind in ("hybrid", "table")],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"bag": report["bag"], "models": [
        {"model": row["model"], "output_count": row["output_count"],
         "speed_raw": row["speed_raw"],
         "position_quality": row["position"]["quality"],
         "endpoint": row["position"]["endpoint"],
         "drive_model_tick_counts": row["drive_model_tick_counts"]}
        for row in report["models"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
