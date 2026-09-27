#!/usr/bin/env python3
"""Reproducible full ROS replay on isolated topics; GNSS is monitor-only after init."""
import argparse
import bisect
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import uuid

import yaml
import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor
from tram_odometry.route_localizer import RouteLocalizer
from accuracy_metrics import evaluate_position


def stamp(m):
    return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec


def stats(values):
    if not values:
        return None
    return {"n": len(values), "mae": statistics.mean(abs(x) for x in values),
            "rmse": math.sqrt(statistics.mean(x*x for x in values)),
            "bias": statistics.mean(values)}


def percentiles(values):
    if not values:
        return None
    values = sorted(values)
    return {"n": len(values), "p50": values[len(values)//2],
            "p95": values[min(len(values)-1, int(len(values)*.95))],
            "p99": values[min(len(values)-1, int(len(values)*.99))],
            "max": values[-1]}


def nearest_index(values, key, tolerance=50_000_000):
    i = bisect.bisect_left(values, key)
    candidates = [k for k in (i-1, i) if 0 <= k < len(values)]
    if not candidates:
        return None
    k = min(candidates, key=lambda j: abs(values[j]-key))
    return k if abs(values[k]-key) <= tolerance else None


class Monitor(Node):
    def __init__(self, prefix):
        super().__init__("team_bag_monitor_" + uuid.uuid4().hex)
        self.velocity, self.position, self.gnss, self.fix, self.rover_fix, self.diagnostics = [], [], [], [], [], []
        self.commands = {}
        self.prefix = prefix
        self.create_subscription(VelocitySensor, prefix + "/result/velocity", self.on_velocity, 10000)
        self.create_subscription(Odometry, prefix + "/result/position",
                                 lambda m: self.position.append((stamp(m), m, time.monotonic())), 10000)
        self.create_subscription(TwistStamped, prefix + "/sensing/gnss/master/vel",
                                 lambda m: self.gnss.append((stamp(m), math.sqrt(sum(
                                     x*x for x in (m.twist.linear.x,m.twist.linear.y,m.twist.linear.z))))), 10000)
        self.create_subscription(NavSatFix, prefix + "/sensing/gnss/master/fix",
                                 lambda m: self.fix.append((stamp(m),m)), 10000)
        self.create_subscription(NavSatFix, prefix + "/sensing/gnss/rover/fix",
                                 lambda m: self.rover_fix.append((stamp(m),m)), 10000)
        self.create_subscription(DriverControllerCommand, prefix + "/vehicle/driver_position_cmd",
                                 lambda m: self.commands.setdefault(stamp(m),time.monotonic()), 10000)
        self.create_subscription(DiagnosticArray, prefix + "/result/diagnostics",
                                 self.on_diagnostic, 1000)

    def on_velocity(self, m):
        self.velocity.append((stamp(m), m.velocity, time.monotonic()))

    def on_diagnostic(self, m):
        if m.status:
            self.diagnostics.append({kv.key: kv.value for kv in m.status[0].values})


def stop(p):
    if p is not None and p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=3)


def read_usage(pid):
    try:
        fields = Path("/proc/%s/stat" % pid).read_text().split()
        ticks = int(fields[13])+int(fields[14])
        status = Path("/proc/%s/status" % pid).read_text().splitlines()
        memory = {line.split(":")[0]: int(line.split()[1])
                  for line in status if line.startswith(("VmRSS:", "VmHWM:"))}
        return ticks / os.sysconf("SC_CLK_TCK"), memory
    except (OSError, ValueError):
        return None


def summarize(m, args, usage, elapsed, returncode):
    velocity = sorted(m.velocity)
    times = [v[0] for v in velocity]
    position = sorted(m.position, key=lambda p:p[0])
    ptimes = [p[0] for p in position]
    errors = []
    for ts, gt in m.gnss:
        k = nearest_index(times,ts)
        if k is not None:
            errors.append(velocity[k][1]-gt)
    wall = sorted(v[2] for v in m.velocity)
    # Exclude the buffered header-history burst at startup from nominal wall Hz.
    steady = [v for v in wall if wall and v >= wall[0]+2]
    poswall = sorted(p[2] for p in m.position)
    psteady = [v for v in poswall if poswall and v >= poswall[0]+2]
    def wall_stats(t):
        gaps = [(b-a)*1000 for a,b in zip(t,t[1:])]
        return {"hz": (len(t)-1)/(t[-1]-t[0]) if len(t)>1 else None,
                "gap_ms": percentiles(gaps)}
    velocity_rx = {ts: received for ts, _, received in m.velocity}
    position_by_receive = sorted(
        (p for p in m.position if poswall and p[2] >= poswall[0]+2),
        key=lambda p:p[2])
    worst_position_gap = None
    if len(position_by_receive) > 1:
        before, after = max(zip(position_by_receive, position_by_receive[1:]),
                            key=lambda pair:pair[1][2]-pair[0][2])
        before_velocity = velocity_rx.get(before[0])
        after_velocity = velocity_rx.get(after[0])
        worst_position_gap = {
            "previous_stamp_ns": before[0], "current_stamp_ns": after[0],
            "position_receive_gap_ms": (after[2]-before[2])*1000,
            "state_stamp_gap_ms": (after[0]-before[0])/1e6,
            "paired_velocity_receive_gap_ms":
                (after_velocity-before_velocity)*1000
                if before_velocity is not None and after_velocity is not None else None,
            "previous_position_after_velocity_ms":
                (before[2]-before_velocity)*1000 if before_velocity is not None else None,
            "current_position_after_velocity_ms":
                (after[2]-after_velocity)*1000 if after_velocity is not None else None,
        }
    rx_by_stamp = {v[0]:v[2] for v in m.velocity}
    latency = [(rx_by_stamp[s]-t)*1000 for s,t in m.commands.items() if s in rx_by_stamp]
    positive = [t for t in latency if t>=0]
    resource = {
        "cpu_average_one_core_percent": usage[-1][1]/elapsed*100 if usage else None,
        "rss_peak_sampled_kib": max((u[2].get("VmRSS",0) for u in usage),default=0),
        "rss_hwm_kib": max((u[2].get("VmHWM",0) for u in usage),default=0),
    }
    core_rates = [(b[1]-a[1])/(b[0]-a[0])*100 for a,b in zip(usage,usage[1:]) if b[0]>a[0]]
    resource["cpu_peak_sampled_one_core_percent"] = max(core_rates,default=0)
    frames = sorted(set(p[1].header.frame_id for p in position))
    localizer = RouteLocalizer.load(Path(get_package_share_directory("tram_odometry")) /
                                   "config/route_bundle.json")
    last_diagnostic = m.diagnostics[-1] if m.diagnostics else {}
    anchor_value = last_diagnostic.get("last_accepted_fix_stamp_ns")
    anchor_ns = int(anchor_value) if anchor_value and anchor_value != "0" else None
    distance_value = last_diagnostic.get("distance_since_correction_m")
    estimated_distance = float(distance_value) if distance_value not in (None, "None") else None
    position_accuracy = evaluate_position(
        [(ts, msg.pose.pose.position.x, msg.pose.pose.position.y,
          2 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
          msg.header.frame_id) for ts, msg, _ in m.position],
        [(ts, msg.latitude, msg.longitude, msg.altitude, msg.status.status)
         for ts, msg in m.fix],
        localizer.calibration.gnss_to_map,
        anchor_ns=anchor_ns, estimated_distance_m=estimated_distance)
    raw_xy = position_accuracy["raw"]["xy_m"]
    map_xy = ({"n": raw_xy["n"], "mae": raw_xy["mae_m"],
               "rmse": raw_xy["rmse_m"], "bias": raw_xy["mean_m"]}
              if raw_xy is not None else None)
    stamp_gaps = [(b-a)/1e6 for a,b in zip(times,times[1:])]
    result = {
        "bag":str(args.bag), "rate":args.rate, "drive_model_kind":args.drive_model,
        "complete_replay":returncode==0,
        "elapsed_wall_s":elapsed,
        "velocity_count":len(velocity),"position_count":len(position),
        "command_count":len(m.commands),"gnss_velocity_count":len(m.gnss),
        "rover_fix_count":len(m.rover_fix),
        "paired_output_stamps":len(set(times)&set(ptimes)),
        "gnss_paired_within_50ms":len(errors),
        "speed_error_mps":stats(errors),
        "raw_map_xy_error_m":map_xy,
        "map_xy_note":"Raw XY distance after fixed bundle GNSS-to-map transform; no quality mask; not judge-frame validation.",
        "position_accuracy":position_accuracy,
        "velocity_steady_wall":wall_stats(steady),"position_steady_wall":wall_stats(psteady),
        "worst_position_gap_context":worst_position_gap,
        "state_stamp_gaps_ms":percentiles(stamp_gaps),
        "observer_exact_command_latency_ms":percentiles(positive),
        "observer_command_stamp_matches":len(latency),
        "observer_negative_latencies":sum(t<0 for t in latency),
        "latency_note":"Independent subscriber observation for exact command stamps; negative values reflect callback ordering; not a complete internal timing profile.",
        "resources":resource,"frames":frames,
        "diagnostics_count":len(m.diagnostics),
        "drive_model_source_counts":{name:sum(d.get("drive_model_source")==name for d in m.diagnostics)
            for name in sorted({d.get("drive_model_source") for d in m.diagnostics if d.get("drive_model_source")})},
        "drive_model_tick_counts_at_last_diagnostic":json.loads(last_diagnostic["drive_model_tick_counts_json"])
            if "drive_model_tick_counts_json" in last_diagnostic else None,
        "max_drive_model_memory_samples":max((int(d.get("drive_model_memory_samples",0))
            for d in m.diagnostics),default=0),
        "trust_wheels_counts":{v:sum(d.get("trust_wheels")==v for d in m.diagnostics)
            for v in sorted({d.get("trust_wheels") for d in m.diagnostics if d.get("trust_wheels")})},
        "slip_diagnostic_count":sum(d.get("slip_flag")=="True" for d in m.diagnostics),
        "drive_memory_sample_counts":{str(v):sum(d.get("drive_model_memory_samples")==str(v) for d in m.diagnostics)
            for v in range(12)},
        "last_diagnostics":m.diagnostics[-1] if m.diagnostics else None,
        "sampled_callback_work_ms":percentiles([float(d["callback_work_ms"]) for d in m.diagnostics if "callback_work_ms" in d]),
        "all_outputs_finite":all(math.isfinite(v[1]) for v in velocity) and all(
            all(math.isfinite(x) for x in (p[1].pose.pose.position.x,p[1].pose.pose.position.y,
                                          p[1].pose.pose.position.z)) for p in position),
        "output_stamps_strictly_increasing":all(b[0]>a[0] for a,b in zip(m.velocity,m.velocity[1:])),
    }
    meta=yaml.safe_load((args.bag/"metadata.yaml").read_text())["rosbag2_bagfile_information"]
    result["expected_rover_fix_count"]=next((x["message_count"] for x in meta["topics_with_message_count"]
        if x["topic_metadata"]["name"]=="/sensing/gnss/rover/fix"),0)
    result["expected_gnss_velocity_count"]=next((x["message_count"] for x in meta["topics_with_message_count"]
        if x["topic_metadata"]["name"]=="/sensing/gnss/master/vel"),0)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("bag",type=Path)
    parser.add_argument("--rate",type=float,default=1.0)
    parser.add_argument("--drive-model",choices=("hybrid","table"),default="hybrid")
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    args.bag=args.bag.resolve()
    if args.rate<=0:parser.error("rate must be positive")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    meta=yaml.safe_load((args.bag/"metadata.yaml").read_text())["rosbag2_bagfile_information"]
    duration=meta["duration"]["nanoseconds"]/1e9
    prefix="/replay_"+uuid.uuid4().hex
    topics=["/vehicle/front_bogie_velocity","/vehicle/rear_bogie_velocity",
            "/vehicle/driver_position_cmd","/sensing/gnss/master/fix","/sensing/gnss/rover/fix","/sensing/gnss/master/vel"]
    mapping={t:prefix+t for t in topics+["/result/velocity","/result/position","/result/diagnostics","/clock"]}
    rclpy.init(args=[])
    m=Monitor(prefix)
    worker=player=None
    logs=[]
    usage=[]
    try:
        executable=Path(get_package_prefix("tram_odometry"))/"lib/tram_odometry/odometry_node"
        node_log=args.report.with_suffix(".node.log").open("w")
        player_log=args.report.with_suffix(".player.log").open("w")
        logs=[node_log,player_log]
        argv=[str(executable),"--ros-args","-p","use_sim_time:=true",
              "-p","drive_model_kind:="+args.drive_model]
        for old,new in mapping.items():argv+=["-r",old+":="+new]
        began=time.monotonic()
        worker=subprocess.Popen(argv,stdout=node_log,stderr=subprocess.STDOUT,
                                env=dict(os.environ,ROS_LOG_DIR=str(args.report.parent/"ros_logs")))
        deadline=time.monotonic()+10
        while m.count_publishers(prefix+"/result/velocity")<1 or len(
                m.get_subscriptions_info_by_topic(prefix+"/vehicle/driver_position_cmd"))<2:
            assert worker.poll() is None,"Node exited: "+args.report.with_suffix(".node.log").read_text()
            if time.monotonic()>deadline:raise RuntimeError("Node discovery timeout")
            rclpy.spin_once(m,timeout_sec=.02)
        player=subprocess.Popen(["ros2","bag","play",str(args.bag),"--clock","100",
             "--rate",str(args.rate),"--delay","3","--disable-keyboard-controls",
             "--topics",*topics,"--remap",*[old+":="+new for old,new in mapping.items()]],
             stdout=player_log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+duration/args.rate+25
        last_sample=0
        last_progress=began
        while player.poll() is None:
            assert worker.poll() is None,"Node exited during replay"
            if time.monotonic()>deadline:raise RuntimeError("Replay exceeded expected duration")
            rclpy.spin_once(m,timeout_sec=.01)
            now=time.monotonic()
            if now-last_progress>=60:
                print(f"[validate_bag] wall {now-began:.0f}s / about {duration/args.rate+3:.0f}s; outputs {len(m.velocity)}/{len(m.position)}", flush=True)
                last_progress=now
            if now-last_sample>=.25:
                u=read_usage(worker.pid)
                if u:usage.append((now,*u))
                last_sample=now
        until=time.monotonic()+.5
        while time.monotonic()<until:rclpy.spin_once(m,timeout_sec=.01)
        result=summarize(m,args,usage,time.monotonic()-began,player.returncode)
        args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(result,indent=2,allow_nan=False),flush=True)
        assert result["complete_replay"],"Bag player exited unsuccessfully"
        assert result["velocity_count"]>0 and result["position_count"]>0,"No outputs"
        assert result["expected_rover_fix_count"]==0 or result["rover_fix_count"]>0,"No rover GNSS received"
        assert result["all_outputs_finite"] and result["output_stamps_strictly_increasing"]
    finally:
        stop(player)
        stop(worker)
        for log in logs:log.close()
        m.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()
