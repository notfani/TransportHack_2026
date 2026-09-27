"""Accepted composition; the core estimate is read-only to this module."""
from .base import InitializerBase, InitializerConfig
from .heading import HeadingFusion
from .refinement import InitialRefinementMixin

class InitialSnapshotMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.initial_anchor_measurement=None

    def _accept_fix(self,stamp,source,xyz,state,reason):
        before=self.counts['accepted_fix']
        result=super()._accept_fix(stamp,source,xyz,state,reason)
        if self.initial_anchor_measurement is None and self.counts['accepted_fix']>before and reason.startswith('initial_'):
            self.initial_anchor_measurement=(stamp,source,tuple(xyz),self.anchor)
        return result

class FreezeAfterAnchor:
    def _position(self, stamp, source, payload, state):
        if self.anchor is not None:
            self._reject('fix', source, 'initialization_complete')
            return
        return super()._position(stamp, source, payload, state)

class PositionInitializer(InitialSnapshotMixin, HeadingFusion, InitialRefinementMixin, FreezeAfterAnchor, InitializerBase):
    pass
