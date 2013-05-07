# This program is public domain
# Author Paul Kienzle
"""
Layer models for polymer systems.

Analytic Self-consistent Field (SCF) profile [#Zhulina]_ [#Karima]_

layer = TetheredPolymer(polymer,solvent,head,tail,power,

.. [#Zhulina] Zhulina, EB; Borisov, OV; Pryamitsyn, VA; Birshtein, TM (1991)
   "Coil-Globule Type Transitions in Polymers. 1. Collapse of Layers
   of Grafted Polymer Chains", Macromolecules 24, 140-149.

.. [#Karima] Karima, A; Douglas, JF; Horkay, F; Fetters, LJ; Satija, SK (1996)
   "Comparative swelling of gels and polymer brush layers",
   Physica B 221, 331-336. `<DOI: 10.1016/0921-4526(95)00946-9>`_
"""
from __future__ import division
#__all__ = ["TetheredPolymer","VolumeProfile","layer_thickness"]
import inspect
import numpy
from numpy import real, imag, exp
from bumps.parameter import Parameter

from .model import Layer
from . import util


class PolymerBrush(Layer):
    r"""
    Polymer brushes in a solvent

    Parameters:

        *thickness* the thickness of the solvent layer
        *interface* the roughness of the solvent surface
        *polymer* the polymer material
        *solvent* the solvent material or vacuum
        *base_vf* volume fraction (%) of the polymer brush at the interface
        *base* the thickness of the brush interface (A)
        *length* the length of the brush above the interface (A)
        *power* the rate of brush thinning
        *sigma* rms brush roughness (A)

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    These parameters combine in the following profile formula:

    .. math::

        V(z) &= \left\{
          \begin{array}{ll}
            V_o                        & \mbox{if } z <= z_o \\
            V_o (1 - ((z-z_o)/L)^2)^p  & \mbox{if } z_o < z < z_o + L \\
            0                          & \mbox{if } z >= z_o + L
          \end{array}
        \right. \\
        V_\sigma(z)
           &= V(z) \star
                 \frac{e^{-\frac{1}{2}(z/\sigma)^2}}{\sqrt{2\pi\sigma^2}} \\
        \rho(z) &= \rho_p V_\sigma(z) + \rho_s (1-V_\sigma(z))

    where $V_\sigma(z)$ is volume fraction convoluted with brush
    roughness $\sigma$ and $\rho(z)$ is the complex scattering
    length density of the profile.
    """
    def __init__(self, thickness=0, interface=0, name="brush",
                 polymer=None, solvent=None, base_vf=None,
                 base=None, length=None, power=None, sigma=None):
        self.thickness = Parameter.default(thickness, name="brush thickness")
        self.interface = Parameter.default(interface, name="brush interface")
        self.base_vf = Parameter.default(base_vf, name="base_vf")
        self.base = Parameter.default(base, name="base")
        self.length = Parameter.default(length, name="length")
        self.power = Parameter.default(power, name="power")
        self.sigma = Parameter.default(sigma, name="sigma")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name
        # Constraints:
        #   base_vf in [0,1]
        #   base,length,sigma,thickness,interface>0
        #   base+length+3*sigma <= thickness
    def parameters(self):
        return {'solvent':self.solvent.parameters(),
                'polymer':self.polymer.parameters(),
                'base_vf':self.base_vf,
                'base':self.base,
                'length':self.length,
                'power':self.power,
                'sigma':self.sigma,
                }
                
    def profile(self, z):
        base_vf, base, length, power, sigma \
            = [p.value for p in self.base_vf, self.base,
               self.length, self.power, self.sigma]
        base_vf /= 100. # % to fraction
        L0 = base  # if base < thickness else thickness
        L1 = base+length # if base+length < thickness else thickness-L0
        if length == 0:
            v = numpy.ones_like(z)
        else:
            v = (1 - ((z-L0)/(L1-L0))**2)
        v[z<L0] = 1
        v[z>L1] = 0
        brush_profile = base_vf * v**power
        # TODO: we could use Nevot-Croce rather than smearing the profile
        vf = smear(z, brush_profile, sigma)
        return vf

    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw,Pz = slabs.microslabs(thickness)
        # Skip layer if it falls to zero thickness.  This may lead to
        # problems in the fitter, since R(thickness) is non-differentiable
        # at thickness = 0.  "Clip to boundary" range handling will at
        # least allow this point to be found.
        # TODO: consider using this behaviour on all layer types.
        if len(Pw) == 0: return

        Mr,Mi = self.polymer.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try: M,S = M[0],S[0]  # Temporary hack
        except: pass

        vf = self.profile(Pz)
        Pw,vf = util.merge_ends(Pw, vf, tol=1e-3)
        P = M*vf + S*(1-vf)
        Pr, Pi = real(P), imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

def layer_thickness(z):
    """
    Return the thickness of a layer given the microslab z points.

    The z points are at the centers of the bins.  we can use the recurrence
    that boundary b[k] = z[k-1] + (z[k-1] - b[k-1]) to compute the
    total length of the layer.
    """
    return 2 * (numpy.sum(z[-1::-2]) - numpy.sum(z[-2::-2]))

class VolumeProfile(Layer):
    """
    Generic volume profile function

    Parameters:

        *thickness* the thickness of the solvent layer
        *interface* the roughness of the solvent surface
        *material* the polymer material
        *solvent* the solvent material
        *profile* the profile function, suitably parameterized

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    These parameters combine in the following profile formula::

        sld = material.sld * profile + solvent.sld * (1 - profile)

    The profile function takes a depth z and returns a density rho.

    For volume profiles, the returned rho should be the volume fraction
    of the material.  For SLD profiles, rho should be complex scattering
    length density of the material.

    Fitting parameters are the available named arguments to the function.
    The first argument must be *z*, which is the array of depths at which
    the profile is to be evaluated.  It is guaranteed to be increasing, with
    step size 2*z[0].

    Initial values for the function parameters can be given using name=value.
    These values can be scalars or fitting parameters.  The function will
    be called with the current parameter values as arguments.  The layer
    thickness can be computed as :func:`layer_thickness`.

    """
    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(self, thickness=0, interface=0, name="VolumeProfile",
                 material=None, solvent=None, profile=None, **kw):
        if interface != 0: raise NotImplementedError("interface not yet supported")
        if profile is None or material is None or solvent is None:
            raise TypeError("Need polymer, solvent and profile")
        self.name = name
        self.thickness = Parameter.default(thickness, name="solvent thickness")
        self.interface = Parameter.default(interface, name="solvent interface")
        self.profile = profile
        self.solvent = solvent
        self.material = material

        # Query profile function for the list of arguments
        vars = inspect.getargspec(profile)[0]
        #print "vars",vars
        if inspect.ismethod(profile): vars = vars[1:]  # Chop self
        vars = vars[1:]  # Chop z
        #print vars
        unused = [k for k in kw.keys() if k not in vars]
        if len(unused) > 0:
            raise TypeError("Profile got unexpected keyword argument '%s'"%unused[0])
        dups = [k for k in vars
                if k in ('thickness','interface','polymer','solvent','profile')]
        if len(dups) > 0:
            raise TypeError("Profile has conflicting argument '%s'"%dups[0])
        for k in vars: kw.setdefault(k,0)
        for k,v in kw.items():
            setattr(self,k,Parameter.default(v,name=k))

        self._parameters = vars

    def parameters(self):
        P = {'solvent':self.solvent.parameters(),
             'material':self.material.parameters(),
             }
        for k in self._parameters:
            P[k] = getattr(self,k)
        return P

    def render(self, probe, slabs):
        Mr,Mi = self.material.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        #M,S = M[0],S[0]  # Temporary hack
        Pw,Pz = slabs.microslabs(self.thickness.value)
        if len(Pw)== 0: return
        kw = dict((k,getattr(self,k).value) for k in self._parameters)
        #print kw
        phi = self.profile(Pz,**kw)
        try:
            if phi.shape != Pz.shape: raise Exception
        except:
            raise TypeError("profile function '%s' did not return array phi(z)"
                            %self.profile.__name__)
        Pw,phi = util.merge_ends(Pw, phi, tol=1e-3)
        P = M*phi + S*(1-phi)
        slabs.extend(rho = [real(P)], irho = [imag(P)], w = Pw)
        #slabs.interface(self.interface.value)


def smear(z, P, sigma):
    """
    Gaussian smearing

    :Parameters:
        *z* | vector
            equally spaced sample times
        *P* | vector
            sample values
        *sigma* | real
            root-mean-squared convolution width
    :Returns:
        *Ps* | vector
            smeared sample values
    """
    if len(z) < 3: return P
    dz = z[1]-z[0]
    if 3*sigma < dz: return P
    w = int(3*sigma/dz)
    G = exp(-0.5*(numpy.arange(-w,w+1)*(dz/sigma))**2)
    full = numpy.hstack( ([P[0]]*w, P, [P[-1]]*w) )
    return numpy.convolve(full,G/numpy.sum(G),'valid')
