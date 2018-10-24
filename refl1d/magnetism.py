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
of alignment of these moments, and the net direction of the alignment.
The strength of the dipole moment depends on the details of the electronic
structure, so unlike the nuclear scattering potential, it cannot be readily
determined from material composition.  Similarly, net magnetization
depends on the details of the magnetic domains within the material, and
cannot readily be determined from first principles.  The interaction
potential of the net magnetic moment depends on the alignment of the field
with respect to the beam, with a net scattering length density of
:math:`\rho_M \cos(\theta_M)`.  Clearly the scattering measurement will
not be able to distinguish between a reduced net magnetic strength
:math:`\rho_M` and a change in orientation :math:`\theta_M` for an
individual measurement, as should be apparent from the correlated
uncertainty plot produced when both parameters are fit.

Magnetism support is split into two parts: describing the layers
and anchoring them to the structure.
"""
from __future__ import print_function

import numpy
from numpy import inf
from bumps.parameter import Parameter, flatten
from bumps.mono import monospline

from .model import Layer, Stack

class BaseMagnetism(object):
    """
    Magnetic properties of the layer.

    Magnetism is attached to set of nuclear layers by setting the
    *magnetism* property of the first layer to the rendered for the
    magnetic profile, and setting the *magnetism.extent* property to
    say how many layers it extends over.

    *dead_below* and *dead_above* are dead regions within the magnetic
    extent, which allow you to shift the magnetic interfaces relative
    to the nuclear interfaces.

    *interface_below* and *interface_above* are the interface widths
    for the magnetic layer, which default to the interface widths for
    the corresponding nuclear layers if no interfaces are specified.
    For consecutive layers, only *interface_above* is used; any value
    for *interface_below* is ignored.
    """
    def __init__(self, extent=1,
                 dead_below=0, dead_above=0,
                 interface_below=None, interface_above=None,
                 name="LAYER"):
        self.dead_below = Parameter.default(dead_below, limits=(0, inf),
                                            name=name+" deadM below")
        self.dead_above = Parameter.default(dead_above, limits=(0, inf),
                                            name=name+" deadM above")
        if interface_below is not None:
            interface_below = Parameter.default(interface_below, limits=(0, inf),
                                                name=name+" interfaceM below")
        if interface_above is not None:
            interface_above = Parameter.default(interface_above, limits=(0, inf),
                                                name=name+"  interfaceM above")
        self.interface_below = interface_below
        self.interface_above = interface_above
        self.name = name
        self.extent = extent

    def parameters(self):
        return {'dead_below':self.dead_below,
                'dead_above':self.dead_above,
                'interface_below':self.interface_below,
                'interface_above':self.interface_above,
               }

    def set_layer_name(self, name):
        """
        Update the names of the magnetic parameters with the name of the
        layer if it has not already been set.  This is necessary since we don't
        know the layer name until after we have constructed the magnetism object.
        """
        if self.name == "LAYER":
            for p in flatten(self.parameters()):
                p.name = p.name.replace("LAYER", name)
            self.name = name

class Magnetism(BaseMagnetism):
    """
    Region of constant magnetism.
    """
    def __init__(self, rhoM=0, thetaM=270, name="LAYER", **kw):
        BaseMagnetism.__init__(self, name=name, **kw)
        self.rhoM = Parameter.default(rhoM, name=name+" rhoM")
        self.thetaM = Parameter.default(thetaM, limits=(0, 360),
                                        name=name+" thetaM")

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM)
        return parameters

    def render(self, probe, slabs, thickness, anchor, sigma):
        slabs.add_magnetism(anchor=anchor,
                            w=[thickness],
                            rhoM=[self.rhoM.value],
                            thetaM=[self.thetaM.value],
                            sigma=sigma)
    def __str__(self):
        return "magnetism(%g)"%self.rhoM.value

    def __repr__(self):
        return ("Magnetism(rhoM=%g, thetaM=%g)"
                %(self.rhoM.value, self.thetaM.value))

class MagnetismStack(BaseMagnetism):
    """
    Magnetic slabs within a magnetic layer.
    """
    def __init__(self, weight=(), rhoM=(), thetaM=(270,), interfaceM=(0,),
                 name="LAYER", **kw):
        if (len(thetaM) != 1 and len(thetaM) != len(weight)
                and len(rhoM) != 1 and len(rhoM) != len(weight)
                and len(interfaceM) != 1 and len(interfaceM) != len(weight)-1):
            raise ValueError("Must have one rhoM, thetaM and intefaceM for each layer")
        if interfaceM != [0]:
            raise NotImplementedError("Doesn't support magnetic roughness")

        BaseMagnetism.__init__(self, stack=stack, name=name, **kw)
        self.weight = [Parameter.default(v, name=name+" weight[%d]"%i)
                       for i, v in enumerate(weight)]
        self.rhoM = [Parameter.default(v, name=name+" rhoM[%d]"%i)
                     for i, v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name+" thetaM[%d]"%i)
                       for i, v in enumerate(thetaM)]
        self.interfaceM = [Parameter.default(v, name=name+" interfaceM[%d]"%i)
                           for i, v in enumerate(interfaceM)]

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(rhoM=self.rhoM,
                          thetaM=self.thetaM,
                          interfaceM=self.interfaceM,
                          weight=self.weight)
        return parameters

    def render(self, probe, slabs, thickness, anchor, sigma):
        w = numpy.array([p.value for p in self.weight])
        w *= thickness / numpy.sum(w)
        rhoM = [p.value for p in self.rhoM]
        thetaM = [p.value for p in self.thetaM]
        sigmaM = [p.value for p in self.interfaceM]
        if len(rhoM) == 1:
            rhoM = [rhoM[0]]*len(w)
        if len(thetaM) == 1:
            thetaM = [thetaM[0]]*len(w)
        if len(sigmaM) == 1:
            sigmaM = [sigmaM[0]]*(len(w)-1)

        slabs.add_magnetism(anchor=anchor,
                            w=w, rhoM=rhoM, thetaM=thetaM,
                            sigma=sigma)

    def __str__(self):
        return "MagnetismStack(%d)"%(len(self.rhoM))
    def __repr__(self):
        return "MagnetismStack"


class MagnetismTwist(BaseMagnetism):
    """
    Linear change in magnetism throughout layer.
    """
    magnetic = True
    def __init__(self,
                 rhoM=(0, 0), thetaM=(270, 270),
                 name="LAYER", **kw):
        BaseMagnetism.__init__(self, name=name, **kw)
        self.rhoM = [Parameter.default(v, name=name+" rhoM[%d]"%i)
                     for i, v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name+" thetaM[%d]"%i)
                       for i, v in enumerate(thetaM)]

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM)
        return parameters

    def render(self, probe, slabs, thickness, anchor, sigma):
        w, z = slabs.microslabs(thickness)
        rhoM = numpy.linspace(self.rhoM[0].value,
                              self.rhoM[1].value, len(z))
        thetaM = numpy.linspace(self.thetaM[0].value,
                                self.thetaM[1].value, len(z))
        slabs.add_magnetism(anchor=anchor,
                            w=w, rhoM=rhoM, thetaM=thetaM,
                            sigma=sigma)

    def __str__(self):
        return "twist(%g->%g)"%(self.rhoM[0].value, self.rhoM[1].value)
    def __repr__(self):
        return "MagneticTwist"


class FreeMagnetism(BaseMagnetism):
    """
    Spline change in magnetism throughout layer.
    """
    magnetic = True
    def __init__(self, z=(), rhoM=(), thetaM=(),
                 name="LAYER", **kw):
        BaseMagnetism.__init__(self, name=name, **kw)
        def parvec(vector, name, limits):
            return [Parameter.default(p, name=name+"[%d]"%i, limits=limits)
                    for i, p in enumerate(vector)]
        self.rhoM, self.thetaM, self.z \
            = [parvec(v, name+" "+part, limits)
               for v, part, limits
               in zip((rhoM, thetaM, z),
                      ('rhoM', 'thetaM', 'z'),
                      ((0, inf), (0, 360), (0, 1)))
              ]
        if len(self.z) != len(self.rhoM):
            raise ValueError("must have number of intervals dz one less than rhoM")
        if len(self.thetaM) > 0 and len(self.rhoM) != len(self.thetaM):
            raise ValueError("must have one thetaM for each rhoM")

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM, z=self.z)
        return parameters

    def profile(self, Pz, thickness):
        mbelow, tbelow = 0, (self.thetaM[0].value if self.thetaM else 270)
        mabove, tabove = 0, (self.thetaM[-1].value if self.thetaM else 270)
        z = numpy.sort([0]+[p.value for p in self.z]+[1])*thickness

        rhoM = numpy.hstack((mbelow, [p.value for p in self.rhoM], mabove))
        PrhoM = monospline(z, rhoM, Pz)

        if numpy.any(numpy.isnan(PrhoM)):
            print("in mono with bad PrhoM")
            print("z %s"%str(z))
            print("p %s"%str([p.value for p in self.z]))


        if len(self.thetaM) > 1:
            thetaM = numpy.hstack((tbelow, [p.value for p in self.thetaM], tabove))
            PthetaM = monospline(z, thetaM, Pz)
        elif len(self.thetaM) == 1:
            PthetaM = self.thetaM.value * numpy.ones_like(PrhoM)
        else:
            PthetaM = 270*numpy.ones_like(PrhoM)
        return PrhoM, PthetaM

    def render(self, probe, slabs, thickness, anchor, sigma):
        Pw, Pz = slabs.microslabs(thickness)
        rhoM, thetaM = self.profile(Pz, thickness)
        slabs.add_magnetism(anchor=anchor,
                            w=Pw, rhoM=rhoM, thetaM=thetaM,
                            sigma=sigma)

    def __str__(self):
        return "freemag(%d)"%(len(self.rhoM))
    def __repr__(self):
        return "FreeMagnetism"
