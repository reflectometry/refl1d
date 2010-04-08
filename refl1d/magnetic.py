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

class Magnetic(Layer):
    """
    Region of constant magnetism, possibly spanning multiple structural
    layers.
    """
    def __init__(self, stack, rhoM=0, thetaM=270, 
                 dead_below=0, dead_above=0):
        self.stack = stack
        self.rhoM = Parameter.default(rhoM, name=stack.name+" magnetic SLD")
        self.thetaM = Parameter.default(interface, limits=(0,360),
                                        name=stack.name+" magnetic angle")
        self.dead_below = Parameter.default(dead_below,
                                            name=stack.name+" nonmagnetic below")
        self.dead_above = Parameter.default(dead_above,
                                            name=stack.name+" nonmagnetic above")

    def parameters(self):
        return dict(stack=self.stack.parameters,
                    rhoM=self.rhoM, thetaM=self.thetaM,
                    dead_below=self.dead_below, dead_above=self.dead_above)

    def render(self, probe, slabs):
        z = slabs.thickness()
        anchor = len(slabs)
        self.stack.render(probe, slabs)
        dz = slabs.thickness() - z
        wlo, whi = self.dead_below.value, self.dead_above.value
        rhoM, thetaM = self.rhoM.value, self.thetaM.value
        if wlo != 0:
            if whi != 0:
                w = [wlo, dz - (wlo+whi), whi]
                rhoM = [0, rhoM, 0]
                thetaM = [thetaM]*3
            else:
                w = [wlo, dz-wlo]
                rhoM = [0, rhoM]
                thetaM = [thetaM]*2
        elif whi != 0:
            w = [dz-whi, whi]
            rhoM = [rhoM, 0]
            thetaM = [thetaM]*2
        else:
            w = [dz]
            rhoM = [rhoM]
            thetaM = [thetaM]
        slabs.magnetic(anchor, w=w, rhoM=rhoM, thetaM=thetaM)

    def __str__(self):
        return "magnetic("+str(self.stack)+")"
    def __repr__(self):
        return "Magnetic("+repr(self.stack)+")"

class MagneticTwist(Layer):
    """
    Region of constant magnetism, possibly spanning multiple structural
    layers.
    """
    def __init__(self, stack, rhoM=0):
        self.stack = stack
        self.rhoM = Parameter.default(rhoM, name=stack.name+" rhoM")

    def parameters(self):
        return dict(stack=self.stack.parameters,
                    rhoM=self.rhoM,
                    thetaM=self.thetaM)

    def render(self, probe, slabs):
        mark = len(slabs)
        z = slabs.thickness()
        self.stack.render(probe, slabs)
        slabs.magnetic(start, w=[slabs.thickness() - z], 
                       rhoM=[self.rhoM.value], thetaM=[self.thetaM.value])
        
    def __str__(self):
        return "magnetic("+str(self.stack)+")"
    def __repr__(self):
        return "MagneticTwist("+repr(self.stack)+")"
