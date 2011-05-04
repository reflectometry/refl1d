# This program is public domain
# Author: Paul Kienzle
"""
Magnetic modeling for 1-D reflectometry.

Magnetic properties are tied to the structural description of the
but only loosely.

There may be dead regions near the interfaces of magnetic materials.

Magnetic behaviour may be varying in complex ways within and
across structural boundaries.  For example, the ma
Indeed, the pattern may continue
across spacer layers, going to zero in the magnetically dead
region and returning to its long range variation on entry to
the next magnetic layer.  Magnetic multilayers may exhibit complex
magnetism throughout the repeated section while the structural
components are fixed.

The scattering behaviour is dependent upon net field strength relative to
polarization direction.   This arises from three underlying quantities:
the strength of the individual dipole moments in the layer, the degree
of alignment of these moments, and the net direction of the alignment.  The
strength of the dipole moment depends on the details of the electronic
structure, so is not This could in principle be approximated from
the dipole moments of the individual moments
aligned within the sample, then you would see the
If the fields for all carriers are aligned with
the polarization direction, you will see the idealized magnetic scattering
strength
will see the saturated This is determined by the number and strength
of the magnetic 'carriers', the amount of order, and the direct or :math:`\rho_M \cos(\theta_M)`, where
orientation, which leads to over-parameterization in the fits.  The
reflectometry technique is sensitive

Magnetism support is split into two parts: describing the layers
and anchoring them to the structure.
"""

from .model import Layer
from .mystic.parameter import Parameter, Constant

class MagneticSlab(Layer):
    """
    Region of constant magnetism.
    """
    magnetic = True
    def __init__(self, ofset=0, width=0, sigma=0, rhoM=0, thetaM=270,
                 name="magnetic"):
        self.thickness = Constant(0)
        self.interface = Constant(0)
        self.width = Parameter.default(width, limits=(0,inf),
                                       name=name+" width")
        self.offset = Parameter.default(offset, limits=(0,inf),
                                        name=name+" offset")
        self.sigma = Parameter.default(sigma,
                                       limits=(0,inf),
                                       name=name+" sigma")
        self.rhoM = Parameter.default(rhoM, name=name+" SLD")
        self.thetaM = Parameter.default(thetaM, limits=(0,360),
                                        name=name+" angle")

    def parameters(self):
        return dict(width=self.width,
                    offset=self.offset,
                    sigma=self.sigma,
                    rhoM=self.rhoM,
                    thetaM=self.thetaM,
                    )

    def render(self, probe, slabs):
        anchor = slabs.thickness() + self.offset.value
        slabs.add_magnetism(anchor,
                            w=[self.width.value],
                            rho=[self.rhoM.value],
                            theta=[self.thetaM.value],
                            sigma=[self.sigma.value])
    def __str__(self):
        return "magnetic(%g)"%self.rho.value
    def __repr__(self):
        return "Magnetic(rhoM=%g,thetaM=%g)"%(self.rhoM.value,self.thetaM.value)

class MagneticTwist(Layer):
    """
    Linear change in magnetism throughout layer.
    """
    magnetic = True
    def __init__(self, ofset=0, width=0, sigma=0,
                 rhoM=[0,0], thetaM=[270,270],
                 name="twist"):
        self.thickness = Constant(0)
        self.interface = Constant(0)
        self.width = Parameter.default(width, limits=(0,inf),
                                       name=name+" width")
        self.offset = Parameter.default(offset, limits=(0,inf),
                                        name=name+" offset")
        self.sigma = Parameter.default(sigma,
                                       limits=(0,inf),
                                       name=name+" sigma")
        self.rhoM = [Parameter.default(v, name=name+" SLD[%d]"%i)
                     for i,v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name+" angle[%d]"%i)
                       for i,v in enumerate(thetaM)]

    def parameters(self):
        return dict(width=self.width,
                    offset=self.offset,
                    sigma=self.sigma,
                    rhoM=self.rhoM,
                    thetaM=self.thetaM)

    def render(self, probe, slabs):
        w,z = slabs.microslabs(self.width.value)
        rhoM = numpy.linspace(self.rhoM[0].value,self.rhoM[1].value,len(z))
        thetaM = numpy.linspace(self.thetaM[0].value,self.thetaM[1].value,len(z))
        slabs.add_magnetism(w=w,rhoM=rhoM,thetaM=thetaM,
                            sigma=self.sigma.value)

    def __str__(self):
        return "twist(%g->%g)"%(self.rhoM[0].value,self.rhoM[1].value)
    def __repr__(self):
        return "MagneticTwist"
