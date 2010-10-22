"""
.. sidebar:: On this Page
    
        * :class:`Free form <refl1d.freeform.Freeform>`
"""

import numpy
from numpy import inf
from mystic import Parameter as Par, IntegerParameter as IntPar
from .model import Layer
from .bspline import pbs, bspline

#TODO: add left_sld,right_sld to all layers so that fresnel works
#TODO: access left_sld,right_sld so freeform doesn't need left,right
#TODO: restructure to use vector parameters
#TODO: allow the number of layers to be adjusted by the fit
class Freeform(Layer):
    """
    A freeform section of the sample modeled with B-splines.

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    number of control points. The control points can be equally spaced
    in the layers unless rhoz or irhoz are specified. If the z values
    are given, they must be in the range [0,1].  One control point is
    anchored at either end, so there are two fewer z values than controls
    if z values are given.

    Layers have a slope of zero at the ends, so the automatically blend
    with slabs.
    """
    def __init__(self, thickness=0, left=None, right=None,
                 rho=[], irho=[], rhoz=[], irhoz=[], name="Freeform"):
        self.name = name
        self.left, self.right = left,right
        self.thickness = Par.default(thickness, limits=(0,inf),
                                   name=name+" thickness")
        self.rho,self.irho,self.rhoz,self.irhoz \
            = [[Par.default(p,name=name+" [%d] %s"%(i,part),limits=limits)
                for i,p in enumerate(v)]
               for v,part,limits in zip((rho, irho, rhoz, irhoz),
                                        ('rho', 'irho', 'rhoz', 'irhoz'),
                                        ((-inf,inf),(-inf,inf),(0,1),(0,1))
                                        )]
        if len(self.rhoz) > 0 and len(self.rhoz) != len(self.rho):
            raise ValueError("must have one z value for each rho")
        if len(self.irhoz) > 0 and len(self.irhoz) != len(self.irho):
            raise ValueError("must have one z value for each irho")
    def parameters(self):
        return dict(rho=self.rho,
                    rhoz=self.rhoz,
                    irho=self.irho,
                    irhoz=self.irhoz,
                    left=self.left.parameters(),
                    right=self.right.parameters(),
                    thickness=self.thickness)
    def render(self, probe, slabs):
        thickness = self.thickness.value
        left_rho,left_irho = self.left.sld(probe)
        right_rho,right_irho = self.right.sld(probe)
        Pw,Pz = slabs.microslabs(thickness)
        t = Pz/thickness
        Prho = _profile(left_rho, right_rho, self.rho, self.rhoz, t)
        Pirho = _profile(left_irho, right_irho, self.irho, self.irhoz, t)
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)

def _profile(left,right,control,z,t):
    cy = numpy.hstack((left,[p.value for p in control],right))
    if len(z) > 0:
        cx = numpy.hstack((0,[p.value for p in z],1))
        return pbs(cx,cy,t)
    else:
        return bspline(cy,t)
