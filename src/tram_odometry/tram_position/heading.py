"""Receiver fixes supply a causal initial body heading only."""
from collections import OrderedDict
import math
from .base import InitializerBase
from .dual_heading import DualHeadingGate

LEVER=(12.44189824202542,-.039263907394398556,.000800000000005241)

class MasterOnlyFusion(InitializerBase):
    def receive_fix(self,*args,receive_ns=None,**kwargs):
        return super().receive_fix(*args,**kwargs)
    def receive_velocity(self,*args,receive_ns=None,**kwargs):
        return super().receive_velocity(*args,**kwargs)
    def _position(self,stamp,source,payload,state):
        if source!='master':self._reject('fix',source,'source_policy');return
        return super()._position(stamp,source,payload,state)
    def _velocity(self,stamp,source,payload,state,estimate):
        if source!='master':self._reject('velocity',source,'source_policy');return estimate
        return super()._velocity(stamp,source,payload,state,estimate)

class HeadingFusion(MasterOnlyFusion):
    def __init__(self,*args,lever_body_m=LEVER,**kwargs):
        super().__init__(*args,**kwargs)
        self.heading_gate=DualHeadingGate(lever_body_m);self.lever=lever_body_m
        self.receiver_pending=OrderedDict();self.master_payloads=OrderedDict()
        self.heading=None
        self.latest_master_observed_stamp=0
        self.latest_course=None;self.course_pending=OrderedDict()

    def receive_fix(self,stamp_ns,lat,lon,alt,status=0,source='master',covariance_type=0,variance_m2=None,receive_ns=None):
        before=self.counts['received_fix']
        result=super().receive_fix(stamp_ns,lat,lon,alt,status,source,covariance_type,variance_m2)
        if self.counts['received_fix']>before:
            # Metadata belongs exactly to the newly validated queued payload.
            arrival=receive_ns if receive_ns is not None else self.last_stamp_ns or None
            self.receiver_pending[(source,stamp_ns)]=(arrival,lat,lon,alt,status)
            while len(self.receiver_pending)>400:self.receiver_pending.popitem(last=False)
        return result

    def receive_velocity(self,stamp_ns,vx,vy,vz,source='master',variance_m2ps2=None,receive_ns=None):
        before=self.counts['received_velocity']
        result=super().receive_velocity(stamp_ns,vx,vy,vz,source,variance_m2ps2)
        if self.counts['received_velocity']>before and source=='master' and math.hypot(vx,vy)>=1:
            arrival=receive_ns if receive_ns is not None else self.last_stamp_ns or None
            self.course_pending[(source,stamp_ns)]=(arrival,math.atan2(vy,vx)-self.localizer.calibration.yaw_map_to_enu_rad)
            while len(self.course_pending)>400:self.course_pending.popitem(last=False)
        return result

    def _velocity(self,stamp,source,payload,state,estimate):
        # Parent observe has now admitted this timestamp to causal history.
        # A queued/rejected future packet must not evict a current course.
        item=self.course_pending.pop((source,stamp),None)
        if item is not None:
            arrival,course=item
            arrival=self.first_stamp_ns if arrival is None else arrival
            if -100_000_000<=arrival-stamp<=500_000_000 and 0<=self.last_stamp_ns-stamp<=500_000_000:
                if self.latest_course is None or stamp>self.latest_course[0]:self.latest_course=(stamp,course)
        return super()._velocity(stamp,source,payload,state,estimate)

    def _consume_receiver(self,stamp,source,payload):
        item=self.receiver_pending.pop((source,stamp),None)
        if item is None:return
        receive,lat,lon,alt,status=item;receive=self.first_stamp_ns if receive is None else receive
        xyz=self.localizer.calibration.gnss_to_map(lat,lon,alt)
        heading=self.heading_gate.observe(source,stamp,receive,xyz,status>=0)
        if source=='master':
            self.latest_master_observed_stamp=max(self.latest_master_observed_stamp,stamp)
            self.master_payloads[stamp]=payload
            while len(self.master_payloads)>8:self.master_payloads.popitem(last=False)
        if heading is not None:self.heading=heading

    def _try_dual_anchor(self):
        h=self.heading
        if h is None or not 0<=self.last_stamp_ns-h.stamp_ns<=500_000_000:return False
        payload=self.master_payloads.get(h.master_stamp_ns);state=self._at(h.master_stamp_ns)
        if payload is None or state is None:return False
        if self.latest_course is not None and 0<=self.last_stamp_ns-self.latest_course[0]<=500_000_000 and state.speed>=.5:
            delta=h.heading_map_rad-self.latest_course[1]
            if abs(math.atan2(math.sin(delta),math.cos(delta)))>math.pi/3:
                self._reject("fix","master","dual_heading_velocity_conflict");return False
        try:
            anchor=self.localizer.anchor(*payload[:3],route_name=self.c.route_name,
                heading_enu_rad=h.heading_map_rad+self.localizer.calibration.yaw_map_to_enu_rad,
                max_lateral_m=self.c.initial_max_lateral_m)
        except ValueError:return False
        self.anchor=anchor;self.offset_m=anchor.chainage_m-state.distance
        xyz=self.localizer.calibration.gnss_to_map(*payload[:3])
        self._accept_fix(h.master_stamp_ns,'master',xyz,state,'initial_dual_heading')
        self.initial_fixes.clear();return True

    def _initial_anchor(self,stamp,source,payload,state):
        # Existing successful terminal prior/moving-GNSS initialization wins.
        super()._initial_anchor(stamp,source,payload,state)
        if self.anchor is None:self._try_dual_anchor()

    def _position(self,stamp,source,payload,state):
        self._consume_receiver(stamp,source,payload)
        if source=='master':return super()._position(stamp,source,payload,state)
        if self.anchor is None and self._try_dual_anchor():return
        self._reject('fix',source,'heading_only_receiver')

