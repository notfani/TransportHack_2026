"""Causal dual-GNSS body heading. No map-derived or predicted heading input."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DualHeading:
    stamp_ns: int
    heading_map_rad: float
    baseline_m: float
    master_xyz: tuple[float,float,float]
    master_stamp_ns: int


class DualHeadingGate:
    def __init__(self,lever_body_m):
        self.length=math.hypot(lever_body_m[0],lever_body_m[1])
        self.angle=math.atan2(lever_body_m[1],lever_body_m[0])
        self.height=lever_body_m[2]
        self.latest={};self.last_pair=None;self.previous=None;self.count=0
        if not 1<=self.length<=30:raise ValueError('unsupported receiver baseline')

    def observe(self,source,stamp_ns,receive_ns,xyz,valid=True):
        if source not in ('master','rover'):raise ValueError('unknown receiver')
        if not valid or not all(math.isfinite(v) for v in xyz):
            self.latest.pop(source,None);self.previous=None;self.count=0;return None
        if not -100_000_000<=receive_ns-stamp_ns<=500_000_000:return None
        if source in self.latest and stamp_ns<=self.latest[source][0]:return None
        self.latest[source]=(stamp_ns,tuple(xyz))
        if len(self.latest)<2:return None
        mt,m=self.latest['master'];rt,r=self.latest['rover'];key=(mt,rt)
        if key==self.last_pair or abs(mt-rt)>50_000_000:return None
        if receive_ns-min(mt,rt)>500_000_000:return None
        self.last_pair=key;dx,dy,dz=(r[i]-m[i] for i in range(3));length=math.hypot(dx,dy)
        if abs(length-self.length)>1 or abs(dz-self.height)>1:
            self.previous=None;self.count=0;return None
        yaw=math.atan2(dy,dx)-self.angle;stamp=max(mt,rt)
        if self.previous is None:self.count=1
        else:
            old_stamp,old_yaw=self.previous
            change=abs(math.atan2(math.sin(yaw-old_yaw),math.cos(yaw-old_yaw)))
            if not 0<stamp-old_stamp<=500_000_000 or change>.25:self.count=1
            else:self.count+=1
        self.previous=(stamp,yaw)
        if self.count<2:return None
        return DualHeading(stamp,yaw,length,m,mt)
