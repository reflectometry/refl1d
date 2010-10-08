# This program is public domain
# Author Paul Kienzle
"""
Layer models for polymer systems.

Analytic Self-consistent Field (SCF) profile[1,2]

layer = TetheredPolymer(polymer,solvent,head,tail,power,

[1] Zhulina, EB; Borisov, OV; Pryamitsyn, VA; Birshtein, TM (1991)
"Coil-Globule Type Transitions in Polymers. 1. Collapse of Layers
of Grafted Polymer Chains", Macromolecules 24, 140-149.

[2] Karima, A; Douglas, JF; Horkay, F; Fetters, LJ; Satija, SK (1996)
"Comparative swelling of gels and polymer brush layers",
Physica B 221, 331-336 DOI: 10.1016/0921-4526(95)00946-9
"""
from __future__ import division
__all__ = ["TetheredPolymer","VolumeProfile","layer_thickness"]
import inspect
import numpy
from numpy import real, imag, exp
from mystic import Parameter
from .model import Layer


class PolymerBrush(Layer):
    """
    Polymer brushes in a solvent

    Parameters::

        *thickness* the thickness of the solvent layer
        *interface* the roughness of the solvent surface
        *polymer* the polymer material
        *solvent* the solvent material or vacuum
        *base_vf* volume fraction of the polymer brush at the interface
        *base* the thickness of the brush interface
        *length* the length of the brush above the interface
        *power* the rate of brush thinning
        *sigma* brush roughness (rms)

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    These parameters combine in the following profile formula::

        brush(z) = base_vf   for z <= base
                 = base_vf * (1 - ((z-base)/length)**2)**power
                             for base <= z <= base+length
                 = 0         for z >= base+length
        profile(z) = conv(brush(z), gaussian(sigma))
        sld(z) = material.sld * profile(z) + solvent.sld * (1 - profile(z))
    """
    def __init__(self, thickness=0, interface=0,
                 polymer=None, solvent=None, base_vf=None,
                 base=None, length=None, power=None, sigma=None):
        self.thickness = Parameter.default(thickness, name="solvent thickness")
        self.interface = Parameter.default(interface, name="solvent interface")
        self.base_vf = Parameter.default(base_vf, name="base_vf")
        self.base = Parameter.default(base, name="base")
        self.length = Parameter.default(length, name="length")
        self.power = Parameter.default(power, name="power")
        self.sigma = Parameter.default(sigma, name="sigma")
        self.solvent = solvent
        self.polymer = polymer
        # Constraints:
        #   base_vf in [0,1]
        #   base,length,sigma,thickness,interface>0
        #   base+length+3*sigma <= thickness
    def parameters(self):
        return dict(solvent=self.solvent.parameters(),
                    polymer=self.polymer.parameters(),
                    thickness=self.thickness,
                    interface=self.interface,
                    base_vf=self.base_vf,
                    base=self.base,
                    length=self.length,
                    power = self.power,
                    sigma = self.sigma)
    def render(self, probe, slabs):
        thickness, interface, base_vf, base, length, power, sigma \
            = [p.value for p in self.thickness, self.interface,
               self.base_vf, self.base, self.length, self.power, self.sigma]
        base_vf /= 100. # % to fraction
        Mr,Mi = self.polymer.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try: M,S = M[0],S[0]  # Temporary hack
        except: pass
        L0 = base if base < thickness else thickness
        L1 = length if base+length < thickness else thickness-L0

        Pw,Pz = slabs.microslabs(thickness)
        brush_profile = base_vf * (1 - ((Pz-L0)/(L1-L0))**2)**power
        brush_profile[Pz<L0] = base_vf
        brush_profile[Pz>L1] = 0
        # TODO: we could use Nevot-Croce rather than smearing the profile
        vf = smear(Pz, brush_profile, sigma)
        P = M*vf + S*(1-vf)
        Pr, Pi = real(P), imag(P)

        # Optimization: find portion of the profile that is varying
        # Rely on the fact that the profile is monotonic
        delta = 0.001
        if abs(Pr[0]-Pr[-1]) <= delta:
            left, right = 0, 0
        elif Pr[0] > Pr[-1]: # decreasing
            left = len(Pr) - numpy.searchsorted(Pr[::-1],Pr[0]-delta)
            right = len(Pr) - numpy.searchsorted(Pr[::-1],Pr[-1]+delta)
        else: # increasing
            left = numpy.searchsorted(Pr, Pr[0]+delta)
            right = numpy.searchsorted(Pr, Pr[-1]-delta)
        if left > right: raise RuntimeError("broken profile search")

        if left > 0:
            slabs.extend(rho=[Pr[0:1]], irho=[Pi[0:1]],
                         w=[numpy.sum(Pw[:left])])
        if left < right:
            slabs.extend(rho=[Pr[left:right]], irho=[Pi[left:right]],
                         w=Pw[left:right])
        if right < len(P):
            slabs.extend(rho=[Pr[-1:]], irho=[Pi[-1:]],
                         w=[numpy.sum(Pw[:right])],
                         sigma = [interface])



def layer_thickness(z):
    """
    Return the thickness of a layer given the microslab z points.

    The layer is sliced into bins of equal width, with the final
    bin making up the remainder.  The z values given to the profile
    function are the centers of these bins.  Using this, we can
    guess that the total layer thickness will be the following::

         2*z[-1]-z[-2] if len(z) > 0 else 2*z[0]
    """
    2*z[-1]-z[-2] if len(z) > 0 else 2*z[0]

class VolumeProfile(Layer):
    """
    Generic volume profile function

    Parameters::

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
    thickness can be computed as :function:`thickness`(z).
    """
    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(self, thickness=0, interface=0,
                 material=None, solvent=None, profile=None, **kw):
        if profile is None or material is None or solvent is None:
            raise TypeError("Need polymer, solvent and profile")
        self.thickness = Parameter.default(thickness, name="solvent thickness")
        self.interface = Parameter.default(interface, name="solvent interface")
        self.profile = profile
        self.solvent = solvent
        self.material = material

        # Query profile function for the list of arguments
        vars = inspect.getargspec(profile)[0]
        print "vars",vars
        if inspect.ismethod(profile): vars = vars[1:]  # Chop self
        vars = vars[1:]  # Chop z
        print vars
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
        P = dict(solvent=self.solvent.parameters(),
                 material=self.material.parameters(),
                 thickness=self.thickness,
                 interface=self.interface)
        for k in self._parameters:
            P[k] = getattr(self,k)

    def render(self, probe, slabs):
        Mr,Mi,Minc = self.material.sld(probe)
        Sr,Si,Sinc = self.solvent.sld(probe)
        M = Mr + 1j*(Mi+Minc)
        S = Sr + 1j*(Si+Sinc)
        M,S = M[0],S[0]  # Temporary hack
        Pw,Pz = slabs.microslabs(self.thickness.value)
        kw = dict((k,getattr(self,k).value) for k in self._parameters)
        print kw
        phi = self.profile(Pz,**kw)
        try:
            if phi.shape != Pz.shape: raise Exception
        except:
            raise TypeError("profile function '%s' did not return array phi(z)"
                            %self.profile.__name__)

        P = M*phi + S*(1-phi)
        slabs.extend(rho = [real(P)], irho = [imag(P)], w = Pw)
        slabs.interface(self.interface.value)


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
    dz = z[1]-z[0]
    if 3*sigma < dz: return P
    w = int(3*sigma/dz)
    G = exp(-0.5*(numpy.arange(-w,w+1)*(dz/sigma))**2)
    full = numpy.hstack( ([P[0]]*w, P, [P[-1]]*w) )
    return numpy.convolve(full,G/numpy.sum(G),'valid')
