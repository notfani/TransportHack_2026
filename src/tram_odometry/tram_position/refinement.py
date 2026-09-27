"""Refine only the first accepted anchor; later tracking stays frozen."""
from collections import deque
import statistics

class InitialRefinementMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.refinement_samples=deque(maxlen=40)
        self.refinement_start_ns=0
        self.refinement_sample_ns=0
        self.refinement_observed_ns=0
        self.original_offset=None
        self.refinement_closed=False

    def _accept_fix(self,stamp,source,xyz,state,reason):
        result=super()._accept_fix(stamp,source,xyz,state,reason)
        # Dual heading may create a master-reference anchor while processing a
        # rover callback. Initialize from the accepted master stamp, not from
        # the source of the callback that happened to complete the pair.
        if source=='master' and reason.startswith('initial_') and not self.refinement_start_ns:
            self.refinement_start_ns=stamp
            self.original_offset=self.offset_m
            self.refinement_samples.append(self.offset_m)
            self.refinement_sample_ns=stamp
            self.refinement_observed_ns=stamp
        return result

    def _position(self,stamp,source,payload,state):
        if source!='master':return super()._position(stamp,source,payload,state)
        if self.anchor is None:
            return super()._position(stamp,source,payload,state)
        if self.refinement_start_ns and stamp-self.refinement_start_ns>round(self.c.initial_window_s*1e9):
            self.refinement_closed=True
        if self.refinement_closed or not self.refinement_start_ns:
            return super()._position(stamp,source,payload,state)
        if stamp<=max(self.refinement_observed_ns,self.last_observed_fix_stamp_ns,self.last_accepted_fix_stamp_ns):
            self._reject('fix',source,'initial_refinement_observed_time_order');return
        self.refinement_observed_ns=stamp
        # The consumed initial stream is real evidence of availability even if
        # a cluster is not yet accepted. It must not look like a GNSS outage.
        self.last_observed_fix_stamp_ns=max(self.last_observed_fix_stamp_ns,stamp)
        if hasattr(self,'previous_observed_position'):
            self.previous_observed_position=(stamp,state.distance)
        if stamp-self.refinement_sample_ns<100_000_000:
            self._reject('fix',source,'initial_refinement_duplicate_pair');return
        if payload[3]>self.c.position_sigma_m**2:
            self._reject('fix',source,'initial_refinement_large_reported_variance');return
        xyz=self.localizer.calibration.gnss_to_map(*payload[:3])
        route=next(r for r in self.localizer.routes if r.name==self.anchor.route_name)
        candidates=route.projection_candidates(xyz[0],xyz[1],self.c.initial_max_lateral_m)
        if not candidates:
            self._reject('fix',source,'initial_refinement_off_map');return
        expected=state.distance+self.original_offset
        candidates.sort(key=lambda c:abs(c[0]-expected))
        s,lateral2,_=candidates[0]
        if self._endpoint_outside(route,s,xyz):
            self._reject('fix',source,'outside_map');return
        if len(candidates)>1 and abs(abs(candidates[1][0]-expected)-abs(s-expected))<3:
            self._reject('fix',source,'initial_refinement_branch_ambiguous');return
        self.refinement_sample_ns=stamp
        self.refinement_samples.append(s-state.distance)
        offsets=sorted(self.refinement_samples)
        clusters=[[v for v in offsets if x<=v<=x+6.] for x in offsets]
        cluster=max(clusters,key=lambda a:(len(a),-(a[-1]-a[0])))
        if len(cluster)<3:
            self._reject('fix',source,'initial_refinement_quorum');return
        target=statistics.median(cluster)
        if abs(target-self.original_offset)>30:
            self._reject('fix',source,'initial_refinement_large_jump');return
        self.last_innovation_m=target-self.offset_m
        self.position_correction_m+=self.last_innovation_m
        self.offset_m=target
        self._accept_fix(stamp,source,xyz,state,'initial_window_refinement')
