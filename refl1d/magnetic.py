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
import numpy
from numpy import inf
from .mystic.parameter import Parameter, Constant
from .model import Layer, Stack
from .mono import monospline

class MagneticLayer(Layer):
    magnetic = True
    def __init__(self, stack=None,
                 dead_below=0, dead_above=0,
                 interface_below=None, interface_above=None,
                 name="magnetic"):
        self.stack = Stack(stack)
        self.dead_below = Parameter.default(dead_below, limits=(0,inf),
                                            name=name+" dead below")
        self.dead_above = Parameter.default(dead_above, limits=(0,inf),
                                            name=name+" dead above")
        self.interface_below = interface_below
        self.interface_above = interface_above
        self.name = name
    def parameters(self):
        return dict(stack=self.stack.parameters(),
                    dead_below=self.dead_below,
                    dead_above=self.dead_above,
                    interface_below=self.interface_below,
                    interface_above=self.interface_above)
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
                   if self.interface_below else slabs.sigma[-1])
        self.stack.render(probe, slabs)
        s_above = (self.interface_above.value
                   if self.interface_above else slabs.sigma[-1])
        return anchor, (s_below, s_above)

    @property
    def thicknessM(self):
        return (self.stack.thickness
                - (self.dead_below.value+self.dead_above.value))

class MagneticSlab(MagneticLayer):
    """
    Region of constant magnetism.
    """
    def __init__(self, stack, rhoM=0, thetaM=270, name="magnetic", **kw):
        MagneticLayer.__init__(self, stack=stack, name=name, **kw)
        self.rhoM = Parameter.default(rhoM, name=name+" SLD")
        self.thetaM = Parameter.default(thetaM, limits=(0,360),
                                        name=name+" angle")

    def parameters(self):
        parameters = MagneticLayer.parameters(self)
        parameters.update(rhoM=self.rhoM,
                          thetaM=self.thetaM)
        return parameters

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
        return ("Magnetic(rhoM=%g,thetaM=%g)"
                %(self.rhoM.value,self.thetaM.value))

class MagneticTwist(MagneticLayer):
    """
    Linear change in magnetism throughout layer.
    """
    magnetic = True
    def __init__(self, stack,
                 rhoM=[0,0], thetaM=[270,270],
                 name="twist", **kw):
        MagneticLayer.__init__(self, stack=stack, name=name, **kw)
        self.rhoM = [Parameter.default(v, name=name+" SLD[%d]"%i)
                     for i,v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name+" angle[%d]"%i)
                       for i,v in enumerate(thetaM)]

    def parameters(self):
        parameters = MagneticLayer.parameters(self)
        parameters.update(rhoM=self.rhoM,
                          thetaM=self.thetaM)
        return parameters

    def render(self, probe, slabs):
        anchor, sigma = self.render_stack(probe, slabs)
        w,z = slabs.microslabs(self.thicknessM)
        rhoM = numpy.linspace(self.rhoM[0].value,
                              self.rhoM[1].value,len(z))
        thetaM = numpy.linspace(self.thetaM[0].value,
                                self.thetaM[1].value,len(z))
        slabs.add_magnetism(anchor=anchor,
                            w=w,rhoM=rhoM,thetaM=thetaM,
                            sigma=sigma)

    def __str__(self):
        return "twist(%g->%g)"%(self.rhoM[0].value,self.rhoM[1].value)
    def __repr__(self):
        return "MagneticTwist"


class FreeMagnetic(MagneticLayer):
    """
    Linear change in magnetism throughout layer.
    """
    magnetic = True
    def __init__(self, stack, z = [], rhoM = [], thetaM = [],
                 name="freemag", **kw):
        MagneticLayer.__init__(self, stack=stack, name=name, **kw)
        def parvec(vector,name,limits):
            return [Parameter.default(p,name=name+"[%d]"%i,limits=limits)
                    for i,p in enumerate(vector)]
        self.rhoM, self.thetaM, self.z \
            = [parvec(v,name+" "+part,limits)
               for v,part,limits in zip((rhoM, thetaM, z),
                                        ('rhoM', 'angle', 'z'),
                                        ((0,inf),(0,360),(0,1))
                                        )]
        if len(self.z) != len(self.rhoM):
            raise ValueError("must have number of intervals dz one less than rhoM")
        if len(self.thetaM) > 0 and len(self.rhoM) != len(self.thetaM):
            raise ValueError("must have one thetaM for each rhoM")

    def parameters(self):
        parameters = MagneticLayer.parameters(self)
        parameters.update(rhoM=self.rhoM,
                          thetaM=self.thetaM,
                          z=self.z)
        return parameters

    def profile(self, Pz):
        thickness = self.thickness.value
        mbelow,tbelow = 0,(self.thetaM[0].value if self.thetaM else 270)
        mabove,tabove = 0,(self.thetaM[-1].value if self.thetaM else 270)
        z = numpy.sort([0]+[p.value for p in self.z]+[1])*thickness

        rhoM = numpy.hstack((mbelow, [p.value for p in self.rhoM], mabove))
        PrhoM = monospline(z, rhoM, Pz)

        if numpy.any(numpy.isnan(PrhoM)):
            print "in mono"
            print "z",z
            print "p",[p.value for p in self.z]


        if len(self.thetaM)>0:
            thetaM = numpy.hstack((tbelow, [p.value for p in self.thetaM], tabove))
            PthetaM = monospline(z, thetaM, Pz)
        else:
            PthetaM = 270*numpy.ones_like(PrhoM)
        return PrhoM,PthetaM

    def render(self, probe, slabs):
        anchor, sigma = self.render_stack(probe, slabs)
        Pw,Pz = slabs.microslabs(self.thicknessM)
        rhoM,thetaM = self.profile(Pz)
        slabs.add_magnetism(anchor=anchor,
                            w=Pw,rhoM=rhoM,thetaM=thetaM,
                            sigma=sigma)

    def __str__(self):
        return "freemag(%d)"%(len(self.rhoM))
    def __repr__(self):
        return "FreeMagnetic"
