# This program is public domain
# Author: Paul Kienzle
r"""
Magnetic modeling for 1-D reflectometry.

** Deprecated ** use magnetism to set magnetism on a nuclear layer
rather than creating a magnetic layer with a nuclear layer stack
underneath.

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

import numpy as np
from numpy import inf
from bumps.parameter import Parameter, to_dict
from bumps.mono import monospline

from .model import Layer, Stack
from .reflectivity import BASE_GUIDE_ANGLE as DEFAULT_THETA_M

class MagneticLayer(Layer):
    @property
    def ismagnetic(self):
        return True

    def __init__(self, stack=None,
                 dead_below=0, dead_above=0,
                 interface_below=None, interface_above=None,
                 name="magnetic"):
        self.stack = Stack(stack)
        self.dead_below = Parameter.default(dead_below, limits=(0, inf),
                                            name=name+" dead below")
        self.dead_above = Parameter.default(dead_above, limits=(0, inf),
                                            name=name+" dead above")
        self.interface_below = interface_below
        self.interface_above = interface_above
        self.name = name
    def parameters(self):
        return {
            'stack':self.stack.parameters(),
            'dead_below':self.dead_below,
            'dead_above':self.dead_above,
            'interface_below':self.interface_below,
            'interface_above':self.interface_above,
        }
    def to_dict(self):
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'stack':self.stack,
            'dead_below':self.dead_below,
            'dead_above':self.dead_above,
            'interface_below':self.interface_below,
            'interface_above':self.interface_above,
        })
    @property
    def thickness(self):
        """Thickness of the magnetic region"""
        return self.stack.thickness

    def render_stack(self, probe, slabs):
        """
        Render the nuclear sld structure.

        If either the interface below or the interface above is left
        unspecified, the corresponding nuclear interface is used.

        Returns the anchor point in the nuclear structure and interface
        widths at either end of the magnetic slab.
        """
        anchor = slabs.thickness() + self.dead_below.value

        s_below = (self.interface_below.value
                   if self.interface_below else slabs.surface_sigma)
        self.stack.render(probe, slabs)
        s_above = (self.interface_above.value
                   if self.interface_above else slabs.surface_sigma)
        return anchor, (s_below, s_above)

    def penalty(self):
        return self.stack.penalty()

    @property
    def thicknessM(self):
        return (self.stack.thickness
                - (self.dead_below.value+self.dead_above.value))

class MagneticSlab(MagneticLayer):
    """
    Region of constant magnetism.
    """
    def __init__(self, stack, rhoM=0, thetaM=DEFAULT_THETA_M, name="magnetic", **kw):
        MagneticLayer.__init__(self, stack=stack, name=name, **kw)
        self.rhoM = Parameter.default(rhoM, name=name+" SLD")
        self.thetaM = Parameter.default(thetaM, limits=(0, 360),
                                        name=name+" angle")

    def parameters(self):
        parameters = MagneticLayer.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM)
        return parameters
    def to_dict(self):
        ret = MagneticLayer.to_dict(self)
        ret['rhoM'] = to_dict(self.rhoM)
        ret['thetaM'] = to_dict(self.thetaM)
        return ret

    def render(self, probe, slabs):
        anchor, sigma = self.render_stack(probe, slabs)
        slabs.add_magnetism(anchor=anchor,
                            w=[self.thicknessM],
                            rhoM=[self.rhoM.value],
                            thetaM=[self.thetaM.value],
                            sigma=sigma)
    def __str__(self):
        return "magnetic(%g)"%self.rhoM.value
    def __repr__(self):
        return ("Magnetic(rhoM=%g, thetaM=%g)"
                %(self.rhoM.value, self.thetaM.value))

class MagneticStack(MagneticLayer):
    """
    Magnetic slabs within a magnetic layer.
    """
    def __init__(self, stack, weight=(), rhoM=(), thetaM=(DEFAULT_THETA_M,), interfaceM=(0),
                 name="mag. stack", **kw):
        if (len(thetaM) != 1 and len(thetaM) != len(weight)
                and len(rhoM) != 1 and len(rhoM) != len(weight)
                and len(interfaceM) != 1 and len(interfaceM) != len(weight)-1):
            raise ValueError("Must have one rhoM, thetaM and intefaceM for each layer")
        if interfaceM != [0]:
            raise NotImplementedError("Doesn't support magnetic roughness")

        MagneticLayer.__init__(self, stack=stack, name=name, **kw)
        self.weight = [Parameter.default(v, name=name+" weight[%d]"%i)
                       for i, v in enumerate(weight)]
        self.rhoM = [Parameter.default(v, name=name+" rhoM[%d]"%i)
                     for i, v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name+" angle[%d]"%i)
                       for i, v in enumerate(thetaM)]
        self.interfaceM = [Parameter.default(v, name=name+" interface[%d]"%i)
                           for i, v in enumerate(interfaceM)]

    def parameters(self):
        parameters = MagneticLayer.parameters(self)
        parameters.update(rhoM=self.rhoM,
                          thetaM=self.thetaM,
                          interfaceM=self.interfaceM,
                          weight=self.weight)
        return parameters
    def to_dict(self):
        ret = MagneticLayer.to_dict(self)
        ret['rhoM'] = to_dict(self.rhoM)
        ret['thetaM'] = to_dict(self.thetaM)
        ret['interfaceM'] = to_dict(self.interfaceM)
        ret['weight'] = to_dict(self.weight)
        return ret

    def render(self, probe, slabs):
        anchor, sigma = self.render_stack(probe, slabs)
        w = np.array([p.value for p in self.weight])
        w *= self.thicknessM.value / np.sum(w)
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
        return "MagneticStack(%d)"%(len(self.rhoM))
    def __repr__(self):
        return "MagneticStack"


class MagneticTwist(MagneticLayer):
    """
    Linear change in magnetism throughout layer.
    """
    magnetic = True
    def __init__(self, stack, rhoM=(0, 0), thetaM=(DEFAULT_THETA_M, DEFAULT_THETA_M),
                 name="twist", **kw):
        MagneticLayer.__init__(self, stack=stack, name=name, **kw)
        self.rhoM = [Parameter.default(v, name=name+" rhoM[%d]"%i)
                     for i, v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name+" angle[%d]"%i)
                       for i, v in enumerate(thetaM)]

    def parameters(self):
        parameters = MagneticLayer.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM)
        return parameters
    def to_dict(self):
        ret = MagneticLayer.to_dict(self)
        ret['rhoM'] = to_dict(self.rhoM)
        ret['thetaM'] = to_dict(self.thetaM)
        return ret

    def render(self, probe, slabs):
        anchor, sigma = self.render_stack(probe, slabs)
        w, z = slabs.microslabs(self.thicknessM)
        rhoM = np.linspace(self.rhoM[0].value,
                           self.rhoM[1].value, len(z))
        thetaM = np.linspace(self.thetaM[0].value,
                             self.thetaM[1].value, len(z))
        slabs.add_magnetism(anchor=anchor,
                            w=w, rhoM=rhoM, thetaM=thetaM,
                            sigma=sigma)

    def __str__(self):
        return "twist(%g->%g)"%(self.rhoM[0].value, self.rhoM[1].value)
    def __repr__(self):
        return "MagneticTwist"


class FreeMagnetic(MagneticLayer):
    """
    Linear change in magnetism throughout layer.
    """
    magnetic = True
    def __init__(self, stack, z=(), rhoM=(), thetaM=(),
                 name="freemag", **kw):
        MagneticLayer.__init__(self, stack=stack, name=name, **kw)
        def parvec(vector, name, limits):
            return [Parameter.default(p, name=name+"[%d]"%i, limits=limits)
                    for i, p in enumerate(vector)]
        self.rhoM, self.thetaM, self.z \
            = [parvec(v, name+" "+part, limits)
               for v, part, limits
               in zip((rhoM, thetaM, z),
                      ('rhoM', 'angle', 'z'),
                      ((0, inf), (0, 360), (0, 1)))
              ]
        if len(self.z) != len(self.rhoM):
            raise ValueError("must have number of intervals dz one less than rhoM")
        if len(self.thetaM) > 0 and len(self.rhoM) != len(self.thetaM):
            raise ValueError("must have one thetaM for each rhoM")

    def parameters(self):
        parameters = MagneticLayer.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM, z=self.z)
        return parameters
    def to_dict(self):
        ret = MagneticLayer.to_dict(self)
        ret['rhoM'] = to_dict(self.rhoM)
        ret['thetaM'] = to_dict(self.thetaM)
        ret['z'] = to_dict(self.z)
        return ret

    def profile(self, Pz):
        thickness = self.thickness.value
        mbelow, tbelow = 0, (self.thetaM[0].value if self.thetaM else DEFAULT_THETA_M)
        mabove, tabove = 0, (self.thetaM[-1].value if self.thetaM else DEFAULT_THETA_M)
        z = np.sort([0]+[p.value for p in self.z]+[1])*thickness

        rhoM = np.hstack((mbelow, [p.value for p in self.rhoM], mabove))
        PrhoM = monospline(z, rhoM, Pz)

        if np.any(np.isnan(PrhoM)):
            print("in mono")
            print("z %s" % str(z))
            print("p %s" % str([p.value for p in self.z]))


        if len(self.thetaM) > 1:
            thetaM = np.hstack((tbelow, [p.value for p in self.thetaM], tabove))
            PthetaM = monospline(z, thetaM, Pz)
        elif len(self.thetaM) == 1:
            PthetaM = np.full_like(PrhoM, self.thetaM.value)
        else:
            PthetaM = np.full_like(PrhoM, DEFAULT_THETA_M)
        return PrhoM, PthetaM

    def render(self, probe, slabs):
        anchor, sigma = self.render_stack(probe, slabs)
        Pw, Pz = slabs.microslabs(self.thicknessM)
        rhoM, thetaM = self.profile(Pz)
        slabs.add_magnetism(anchor=anchor,
                            w=Pw, rhoM=rhoM, thetaM=thetaM,
                            sigma=sigma)

    def __str__(self):
        return "freemag(%d)"%(len(self.rhoM))

    def __repr__(self):
        return "FreeMagnetic"
