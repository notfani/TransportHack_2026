#!/usr/bin/env python3
"""Causal offline replay with explicit wheel faults; never modifies the source bag."""
import argparse
import bisect
import json
import math
from pathlib import Path
import sqlite3
import statistics

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TwistStamped
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor
from tram_estimator.drive import TableDrive
from tram_odometry.bridge import Bridge, BridgeConfig
from tram_odometry.route_localizer import RouteLocalizer
from validate_bag import stamp, stats, nearest_index

TOPICS = {
    "/vehicle/front_bogie_velocity": ("front", VelocitySensor),
    "/vehicle/rear_bogie_velocity": ("rear", VelocitySensor),
    "/vehicle/driver_position_cmd": ("command", DriverControllerCommand),
    "/sensing/gnss/master/fix": ("fix", NavSatFix),
    "/sensing/gnss/master/vel": ("truth", TwistStamped),
}
SCENARIOS = ("clean", "front_dropout", "both_dropout", "front_scale_1_5",
             "both_scale_1_5", "both_scale_0_3", "both_freeze", "both_ramp_1_5")


def read_events(bag):
    events = []
    for path in sorted(bag.glob("*.db3")):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            sql = ("SELECT messages.timestamp, topics.name, messages.data FROM messages "
                   "JOIN topics ON messages.topic_id=topics.id WHERE topics.name IN (" +
                   ",".join("?" for _ in TOPICS) + ") ORDER BY messages.timestamp,messages.id")
            for received, topic, data in db.execute(sql, list(TOPICS)):
                kind, cls = TOPICS[topic]
                events.append((received, kind, deserialize_message(data, cls)))
    events.sort(key=lambda e: e[0])
    if not events:
        raise ValueError("No supported input messages in SQLite bag")
    return events


def choose_window(events, length):
    origin = events[0][0]
    front = [((t-origin)/1e9, m.velocity/3.6) for t,k,m in events if k=="front"]
    times = [t for t,_ in front]
    duration = (events[-1][0]-origin)/1e9
    # Deterministic first moving interval; only unmodified wheel data choose the window.
    for start in range(30, max(30, int(duration-length-6)), 5):
        a, b = bisect.bisect_left(times,start), bisect.bisect_left(times,start+length)
        values = [v for _,v in front[a:b]]
        if len(values)>=length*5 and min(values)>2 and statistics.mean(values)>=3:
            return float(start), statistics.mean(values)
    raise ValueError("No qualifying moving window; choose another bag")


def replay(events, drive, localizer, scenario, start, length):
    bridge = Bridge(drive,localizer,BridgeConfig())
    origin = events[0][0]
    tick = origin
    output, last, frozen = [], {}, {}
    injected = 0
    def collect(o, clock):
        if o is not None:
            e=o.estimate
            output.append((e.stamp_ns,e.speed_mps,e.distance_m,
                           (clock-origin)/1e9,e.source,e.trust_wheels,e.slip_flag))
    for clock,kind,msg in events:
        while tick<=clock:
            collect(bridge.tick(tick),tick)
            tick+=20_000_000
        elapsed=(clock-origin)/1e9
        active=start<=elapsed<start+length
        if kind in ("front","rear"):
            value=float(msg.velocity)
            affected=active and scenario!="clean" and (not scenario.startswith("front_") or kind=="front")
            if affected:
                injected+=1
                if scenario.endswith("dropout"):
                    continue
                if scenario.endswith("scale_1_5"):
                    value*=1.5
                elif scenario.endswith("scale_0_3"):
                    value*=.3
                elif scenario=="both_freeze":
                    frozen.setdefault(kind,last.get(kind,value))
                    value=frozen[kind]
                elif scenario=="both_ramp_1_5":
                    value*=1+.5*(elapsed-start)/length
            bridge.receive_wheel(kind,stamp(msg),value,clock)
            last[kind]=float(msg.velocity)
        elif kind=="command":
            collect(bridge.receive_command(stamp(msg),int(msg.position),clock),clock)
        elif kind=="fix":
            bridge.receive_fix(stamp(msg),msg.latitude,msg.longitude,msg.altitude,msg.status.status)
        # Ground truth messages are evaluated later, never sent into the estimator.
    return output,injected


def summary(events,output,start,length):
    origin=events[0][0]
    times=[o[0] for o in output]
    windows={"whole": [], "fault": [], "recovery_5s": []}
    for clock,kind,msg in events:
        if kind!="truth":
            continue
        gt=math.sqrt(sum(v*v for v in (msg.twist.linear.x,msg.twist.linear.y,msg.twist.linear.z)))
        if not math.isfinite(gt):
            continue
        i=nearest_index(times,stamp(msg))
        if i is None:
            continue
        error=output[i][1]-gt
        windows["whole"].append(error)
        elapsed=(clock-origin)/1e9
        if start<=elapsed<start+length:
            windows["fault"].append(error)
        if start+length<=elapsed<start+length+5:
            windows["recovery_5s"].append(error)
    affected=[o for o in output if start<=o[3]<start+length]
    return {
        "output_count":len(output),
        "finite":bool(output) and all(math.isfinite(x) for o in output for x in o[1:3]),
        "strictly_increasing":all(b>a for a,b in zip(times,times[1:])),
        "max_state_stamp_gap_ms":max((b-a)/1e6 for a,b in zip(times,times[1:])),
        "speed_error_mps":{k:stats(v) for k,v in windows.items()},
        "fault_slip_fraction":sum(o[6] for o in affected)/len(affected) if affected else None,
        "fault_mean_wheel_trust":statistics.mean(o[5] for o in affected) if affected else None,
        "fault_sources":sorted(set(o[4] for o in affected)),
        "final_distance_m":output[-1][2],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag",type=Path)
    p.add_argument("--report",required=True,type=Path)
    p.add_argument("--duration",type=float,default=10.0)
    args=p.parse_args()
    if not 0<args.duration<=30:
        p.error("duration must be in (0,30]")
    events=read_events(args.bag)
    start,window_speed=choose_window(events,args.duration)
    share=Path(get_package_share_directory("tram_odometry"))
    drive=TableDrive(share/"config/drive_table.json")
    localizer=RouteLocalizer.load(share/"config/route_bundle.json")
    rows={}
    for scenario in SCENARIOS:
        print("Replaying "+scenario,flush=True)
        output,injected=replay(events,drive,localizer,scenario,start,args.duration)
        rows[scenario]=summary(events,output,start,args.duration)
        rows[scenario]["injected_wheel_messages"]=injected
        rows[scenario]["final_distance_difference_from_clean_m"]=(
            rows[scenario]["final_distance_m"]-rows["clean"]["final_distance_m"])
    result={
        "bag":str(args.bag.resolve()),"mode":"offline_causal_bridge",
        "fault_start_receive_offset_s":start,"fault_duration_s":args.duration,
        "window_selection":"First 5-second-grid window after 30s with front speed min>2m/s and mean>=3m/s",
        "clean_front_window_mean_mps":window_speed,
        "scenarios":rows,
        "contract_checks_passed":all(r["finite"] and r["strictly_increasing"] for r in rows.values()),
        "notes":[
            "Uses recorded receive order and simulated 20ms watchdog; not a ROS transport/timing benchmark.",
            "GNSS velocity used only for scoring, with nearest header stamp within 50ms and no GNSS quality mask.",
            "This is a known provided bag, not a new holdout evaluation.",
            "Finite outputs do not imply accurate fallback. Inspect fault and recovery RMSE separately.",
            "Ground-truth GNSS fix is available only through the existing limited startup path.",
        ]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for name,r in rows.items():
        e=r["speed_error_mps"]
        print(name,"fault RMSE",e["fault"]["rmse"] if e["fault"] else None,
              "recovery RMSE",e["recovery_5s"]["rmse"] if e["recovery_5s"] else None,flush=True)
    if not result["contract_checks_passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()