"""Causal adapter with accepted initial localization and explicit map coverage."""
from __future__ import annotations
import math
from dataclasses import replace
from tram_estimator import Drive
from .transport_bridge import BridgeBase, BridgeConfig, Output
from tram_position import PositionInitializer, InitializerConfig


class Bridge(BridgeBase):
    def __init__(self,drive_model,localizer,config=None,estimator_config=None,*,fusion_class=PositionInitializer):
        super().__init__(drive_model,localizer,config,estimator_config)
        self.fusion=fusion_class(localizer,InitializerConfig(route_name=None if self.c.route_name=='auto' else self.c.route_name,
            initial_window_s=self.c.initialization_window_s,initial_tolerance_s=self.c.initial_fix_tolerance_s,
            initial_max_lateral_m=self.c.max_initial_lateral_m,
            initial_heading_min_displacement_m=self.c.initial_heading_min_displacement_m),self.estimator)
        self.initial_map_context=None
        if self.anchor is not None:
            self.fusion.anchor=self.anchor
            self.fusion.offset_m=self.anchor.chainage_m
        self.gnss_closed=self.c.start_mode!='gnss' or self.c.initialization_window_s<=0

    def _receipt_frontier(self,clock_ns):
        # Clock is arrival/replay clock; GNSS headers are in a different domain.
        if clock_ns is not None and self.header_anchor_ns and clock_ns>=self.clock_anchor_ns:
            return self.header_anchor_ns+(clock_ns-self.clock_anchor_ns)
        return self.estimator.stamp_ns or None

    def receive_fix(self,stamp_ns,lat,lon,alt,status=0,*,source='master',covariance_type=0,variance_m2=None,clock_ns=None):
        if self.gnss_closed:
            self.counters['ignored_gnss']+=1
            return
        return self.fusion.receive_fix(stamp_ns,lat,lon,alt,status=status,source=source,covariance_type=covariance_type,
            variance_m2=variance_m2,receive_ns=self._receipt_frontier(clock_ns))

    def receive_velocity(self,stamp_ns,vx,vy,vz,*,source='master',variance_m2ps2=None,clock_ns=None):
        if self.gnss_closed:
            self.counters['ignored_gnss']+=1
            return
        return self.fusion.receive_velocity(stamp_ns,vx,vy,vz,source=source,variance_m2ps2=variance_m2ps2,
            receive_ns=self._receipt_frontier(clock_ns))

    def _advance_pose(self,stamp_ns,trigger):
        if stamp_ns<=self.estimator.stamp_ns:return None
        c=self.c
        stale=not self.command_stamp_ns or (stamp_ns-self.command_stamp_ns)/1e9>c.command_timeout_s
        command=0 if stale else self.command
        previous_speed=self.last_output.estimate.speed_mps if self.last_output else 0.
        if stale and hasattr(self.drive_model, "reset_unavailable"):
            self.drive_model.reset_unavailable()
        dt=(stamp_ns-self.estimator.stamp_ns)/1e9 if self.estimator.stamp_ns else .1
        drive=(Drive(0.,c.missing_command_sigma_mps2) if stale else
            self.drive_model.predict_interval(command,previous_speed,dt)
            if hasattr(self.drive_model,"predict_interval") else
            self.drive_model.predict(command,previous_speed))
        corrected=[self.fusion.correct_wheel(name,getattr(self,name)) for name in ('front','rear')]
        estimate=self.estimator.update(stamp_ns,command,*corrected,drive)
        if hasattr(self.drive_model,"observe_wheels"):
            self.drive_model.observe_wheels(estimate,*corrected)
        fused=self.fusion.observe(estimate,raw_front=self.front,raw_rear=self.rear)
        estimate=fused.estimate
        self.history.append((estimate.stamp_ns,estimate.distance_m))
        self.anchor=fused.anchor
        if self.anchor is not None:self.anchor_distance_m=self.anchor.chainage_m-self.fusion.offset_m
        pose=fused.pose
        valid=pose is not None and pose.region!='outside_map'
        outside=pose.distance_outside_map_m if pose is not None else 0.
        if not valid and self.anchor is not None and fused.chainage_m is not None:
            route=next(r for r in self.localizer.routes if r.name==self.anchor.route_name)
            outside=max(0.,-fused.chainage_m,fused.chainage_m-route.length_m)
        if valid:
            route=next(r for r in self.localizer.routes if r.name==self.anchor.route_name)
            before,after=(route.pose_at(max(0.,min(route.length_m,fused.chainage_m+d))) for d in (-.5,.5))
            delta=(after.x-before.x,after.y-before.y,after.z-before.z)
            length=math.sqrt(sum(v*v for v in delta))
            tangent=tuple(v/length for v in delta) if length>1e-9 else (math.cos(pose.yaw_rad),math.sin(pose.yaw_rad),0.)
            a=c.output_yaw_rad;ca,sa=math.cos(a),math.sin(a)
            xyz=(c.output_x_m+ca*pose.x-sa*pose.y,c.output_y_m+sa*pose.x+ca*pose.y,c.output_z_m+pose.z)
            yaw,frame,region=pose.yaw_rad+a,c.map_frame_id,pose.region
            tangent=(ca*tangent[0]-sa*tangent[1],sa*tangent[0]+ca*tangent[1],tangent[2])
        else:
            yaw,frame=c.start_yaw_rad,c.relative_frame_id
            region='outside_map_relative' if self.anchor is not None else 'relative'
            xyz=(c.start_x_m+estimate.distance_m*math.cos(yaw),c.start_y_m+estimate.distance_m*math.sin(yaw),c.start_z_m)
            tangent=(math.cos(yaw),math.sin(yaw),0.)
        ps=max(0.,estimate.covariance[0][0]);cov=[0.]*36
        for i in range(3):
            for j in range(3):cov[6*i+j]=ps*tangent[i]*tangent[j]
        xy_floor=max(c.map_sigma_xy_m**2,self.anchor.lateral_error_m**2) if valid else 0.
        cov[0]+=xy_floor;cov[7]+=xy_floor
        cov[14]+=c.map_sigma_z_m**2
        cov[21]=cov[28]=1e6;cov[35]=c.map_sigma_yaw_rad**2 if valid else 1e6
        diag=dict(self.counters)
        diag.update(fused.diagnostics)
        self.initialization_reason=self.anchor.selection_reason if self.anchor is not None else self.fusion.last_reason
        diag["drive_model_memory_samples"]=len(getattr(self.drive_model,"history",()))
        diag["drive_model_trusted_age_s"]=round(getattr(getattr(self.drive_model,"state",None),"trusted_age",0.0),3)
        diag.update(trigger=trigger,command_stale=stale,command_used=command,
            route_initialized=self.anchor is not None,absolute_pose_valid=valid,absolute_frame_confirmed=False,
            covariance_calibrated=False,initialization=self.initialization_reason,route_region=region,
            outside_map_m=outside,chainage_m=fused.chainage_m,gnss_initialization_closed=self.gnss_closed,
            source=estimate.source,drive_model_source=getattr(self.drive_model,"source","table"),slip_flag=estimate.slip_flag,trust_wheels=estimate.trust_wheels,reasons='|'.join(estimate.reasons))
        for name in ('front','rear'):
            wheel=getattr(self,name);diag[name+'_age_s']=(stamp_ns-wheel.stamp_ns)/1e9 if wheel is not None else None
        self.last_output=Output(estimate,xyz,yaw,frame,tuple(cov),diag)
        return self.last_output

    def _advance(self,stamp_ns,trigger):
        out=self._advance_pose(stamp_ns,trigger)
        if out is None:return None
        if self.initial_map_context is None and self.anchor is not None:
            snapshot=self.fusion.initial_anchor_measurement
            stamp,source,xyz,a=snapshot if snapshot is not None else (None,None,None,self.anchor)
            route=next(r for r in self.localizer.routes if r.name==a.route_name)
            beyond=0.;endpoint=False
            if xyz is not None and (a.chainage_m<=1e-6 or a.chainage_m>=route.length_m-1e-6):
                endpoint=True
                p,q=route.points[:2] if a.chainage_m<=1e-6 else route.points[-2:]
                origin=p if a.chainage_m<=1e-6 else q
                dx,dy=q[0]-p[0],q[1]-p[1]
                along=((xyz[0]-origin[0])*dx+(xyz[1]-origin[1])*dy)/max(1e-9,math.hypot(dx,dy))
                beyond=max(0.,-along if a.chainage_m<=1e-6 else along)
            self.initial_map_context=dict(initial_map_selection=a.selection_reason,
                initial_anchor_fix_stamp_ns=stamp,initial_anchor_fix_source=source,
                initial_endpoint_projection=endpoint,initial_measurement_beyond_endpoint_m=beyond,
                initial_projection_residual_m=a.lateral_error_m,
                initial_gnss_noise_floor_m=self.fusion.c.position_sigma_m,
                initial_projection_is_uncalibrated_assumption=self.c.start_mode=='gnss')
        diag=dict(out.diagnostics)
        if self.initial_map_context is not None:diag.update(self.initial_map_context)
        self.last_output=replace(out,diagnostics=diag)
        return self.last_output
