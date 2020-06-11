"""
Freeform modeling with B-Splines
"""

import numpy as np
from numpy import inf
from bumps.parameter import Parameter as Par, to_dict
from bumps.bspline import pbs, bspline

from .model import Layer
from . import util

#TODO: add left_sld, right_sld to all layers so that fresnel works
#TODO: access left_sld, right_sld so freeform doesn't need left, right
#TODO: restructure to use vector parameters
#TODO: allow the number of layers to be adjusted by the fit
class FreeLayer(Layer):
    """
    A freeform section of the sample modeled with B-splines.

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    number of control points. The control points can be equally spaced
    in the layers unless rhoz or irhoz are specified. If the z values
    are given, they must be in the range [0, 1].  One control point is
    anchored at either end, so there are two fewer z values than controls
    if z values are given.

    Layers have a slope of zero at the ends, so the automatically blend
    with slabs.
    """
    def __init__(self, thickness=0, left=None, right=None,
                 rho=(), irho=(), rhoz=(), irhoz=(), name="Freeform"):
        self.name = name
        self.left, self.right = left, right
        self.thickness = Par.default(thickness, limits=(0, inf),
                                     name=name+" thickness")
        self.rho, self.irho, self.rhoz, self.irhoz \
            = [[Par.default(p, name=name+" [%d] %s"%(i, part), limits=limits)
                for i, p in enumerate(v)]
               for v, part, limits
               in zip((rho, irho, rhoz, irhoz),
                      ('rho', 'irho', 'rhoz', 'irhoz'),
                      ((-inf, inf), (-inf, inf), (0, 1), (0, 1)))
              ]
        if len(self.rhoz) > 0 and len(self.rhoz) != len(self.rho):
            raise ValueError("must have one z value for each rho")
        if len(self.irhoz) > 0 and len(self.irhoz) != len(self.irho):
            raise ValueError("must have one z value for each irho")
    def parameters(self):
        return {
            'rho': self.rho,
            'rhoz': self.rhoz,
            'irho': self.irho,
            'irhoz': self.irhoz,
            # TODO: left/right pars are already listed in the stack
            'left': self.left.parameters(),
            'right': self.right.parameters(),
        }
    def to_dict(self):
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'parameters': self.parameters(),
        })
    def render(self, probe, slabs):
        thickness = self.thickness.value
        left_rho, left_irho = self.left.sld(probe)
        right_rho, right_irho = self.right.sld(probe)
        Pw, Pz = slabs.microslabs(thickness)
        t = Pz/thickness
        Prho = _profile(left_rho, right_rho, self.rho, self.rhoz, t)
        Pirho = _profile(left_irho, right_irho, self.irho, self.irhoz, t)
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)

class FreeformInterface01(Layer):
    """
    A freeform section of the sample modeled with B-splines.

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    number of control points. The control points can be equally spaced
    in the layers unless rhoz or irhoz are specified. If the z values
    are given, they must be in the range [0, 1].  One control point is
    anchored at either end, so there are two fewer z values than controls
    if z values are given.

    Layers have a slope of zero at the ends, so the automatically blend
    with slabs.
    """
    def __init__(self, thickness=0, interface=0,
                 below=None, above=None,
                 z=None, vf=None, name="Interface"):
        self.name = name
        self.below, self.above = below, above
        self.thickness = Par.default(thickness, limits=(0, inf),
                                     name=name+" thickness")
        self.interface = Par.default(interface, limits=(0, inf),
                                     name=name+" interface")
        if len(z) != len(vf):
            raise ValueError("Need one vf for every z")
        #if len(z) != len(vf)+2:
        #    raise ValueError("Only need vf for interior z, so len(z)=len(vf)+2")
        self.z = [Par.default(p, name=name+" z[%d]"%i, limits=(0, 1))
                  for i, p in enumerate(z)]
        self.vf = [Par.default(p, name=name+" vf[%d]"%i, limits=(0, 1))
                   for i, p in enumerate(vf)]
    def parameters(self):
        return {
            'thickness': self.thickness,
            'interface': self.interface,
            'z': self.z,
            'vf': self.vf,
            'below': self.below.parameters(),
            'above': self.above.parameters(),
        }
    def to_dict(self):
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'parameters': self.parameters(),
        })
    def render(self, probe, slabs):
        thickness = self.thickness.value
        left_rho, left_irho = self.below.sld(probe)
        right_rho, right_irho = self.above.sld(probe)
        z = np.hstack((0, sorted([v.value for v in self.z]), 1))
        vf = np.hstack((0, sorted([v.value for v in self.vf]), 1))
        Pw, Pz = slabs.microslabs(thickness)
        t = Pz/thickness
        offset, profile = pbs(z, vf, t, parametric=False, clamp=True)
        Pw, profile = util.merge_ends(Pw, profile, tol=1e-3)
        Prho = (1-profile)*left_rho + profile*right_rho
        Pirho = (1-profile)*left_irho + profile*right_irho
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)

class FreeInterface(Layer):
    """
    A freeform section of the sample modeled with monotonic splines.

    Layers have a slope of zero at the ends, so the automatically blend
    with slabs.
    """
    def __init__(self, interface=0,
                 below=None, above=None,
                 dz=None, dp=None, name="Interface"):
        self.name = name
        self.below, self.above = below, above
        self.interface = Par.default(interface, limits=(0, inf),
                                     name=name+" interface")

        # Choose reasonable defaults if not given
        if dp is None and dz is None:
            dp = [1]*5
        if dp is None:
            dp = [1]*len(dz)
        if dz is None:
            dz = [10./len(dp)]*len(dp)
        if len(dz) != len(dp):
            raise ValueError("Need one dz for every dp")
        #if len(z) != len(vf)+2:
        #    raise ValueError("Only need vf for interior z, so len(z)=len(vf)+2")
        self.dz = [Par.default(p, name=name+" dz[%d]"%i, limits=(0, inf))
                   for i, p in enumerate(dz)]
        self.dp = [Par.default(p, name=name+" dp[%d]"%i, limits=(0, inf))
                   for i, p in enumerate(dp)]
    def _get_thickness(self):
        w = sum(v.value for v in self.dz)
        return Par(w, name=self.name+" thickness")
    def _set_thickness(self, v):
        if v != 0:
            raise ValueError("thickness cannot be set for FreeInterface")
    thickness = property(_get_thickness, _set_thickness)

    def parameters(self):
        return {
            'interface': self.interface,
            'below': self.below.parameters(),
            'above': self.above.parameters(),
            'dz': self.dz,
            'dp': self.dp,
            }
    def to_dict(self):
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'parameters': self.parameters(),
        })
    def render(self, probe, slabs):
        left_rho, left_irho = self.below.sld(probe)
        right_rho, right_irho = self.above.sld(probe)
        z = np.hstack((0, np.cumsum([v.value for v in self.dz])))
        p = np.hstack((0, np.cumsum([v.value for v in self.dp])))
        if p[-1] == 0:
            p[-1] = 1
        p /= p[-1]
        Pw, Pz = slabs.microslabs(z[-1])
        _, profile = pbs(z, p, Pz, parametric=False, clamp=True)
        profile = np.clip(profile, 0, 1)
        Pw, profile = util.merge_ends(Pw, profile, tol=1e-3)
        Prho = (1-profile)*left_rho + profile*right_rho
        Pirho = (1-profile)*left_irho + profile*right_irho
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)

def _profile(left, right, control, z, t):
    cy = np.hstack((left, [p.value for p in control], right))
    if len(z) > 0:
        cx = np.hstack((0, [p.value for p in z], 1))
        return pbs(cx, cy, t)
    else:
        return bspline(cy, t)
