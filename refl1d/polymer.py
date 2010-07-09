# This program is public domain
# Author Paul Kienzle
"""
Layer models for polymer systems.

Analytic Self-consistent Field (SCF) profile[1,2]

layer = TetheredPolymer(polymer,solvent,head,tail,power,

[1] Zhulina, EB; Borisov, OV; Pryamitsyn, VA; Birshtein, TM (1991)
"Coil-Globule Type Transitions in Polymers. 1. Collapse of Layers
of Grafted Polymer Chains", Macromolecules 24, 140-149.

[2] Kent, M.S.; Majewski, J.; Smith; G.S.; Lee, L.T.; Satija, S (1999)
"Tethered chains in poor solvent conditions: An experimental study
involving Langmuir diblock copolymer monolayers",
J. of Chemical Physics 110(7), 3553-3565    
"""
from __future__ import division
__all__ = ["TetheredPolymer","VolumeProfile","layer_thickness"]
import inspect
from numpy import real, imag
from mystic import Parameter
from .model import Layer


class TetheredPolymer(Layer):
    """
    Tethered polymer in a solvent

    Parameters::

        *thickness* the thickness of the solvent layer
        *interface* the roughness of the solvent surface
        *polymer* the polymer material
        *solvent* the solvent material
        *phi* volume fraction of the polymer head
        *head* the thickness of the polymer head
        *tail* the length of the polymer decay tail
        *power* the rate of decay along the tail

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition 
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    These parameters combine in the following profile formula::
    
        profile(z) = phi   for z <= head
                   = phi * (1 - ((z-head)/tail)**2)**power
                           for head <= z <= head+tail
                   = 0     for z >= head+tail
        sld = material.sld * profile + solvent.sld * (1 - profile)
    """
    def __init__(self, thickness=0, interface=0,
                 polymer=None, solvent=None, phi=None, 
                 head=None, tail=None, Y=None):
        self.thickness = Parameter.default(thickness, name="solvent thickness")
        self.interface = Parameter.default(interface, name="solvent interface")
        self.phi = Parameter.default(phi, name="polymer fraction")
        self.head = Parameter.default(head, name="head thickness")
        self.tail = Parameter.default(tail, name="tail thickness")
        self.Y = Parameter.default(Y, name="tail decay")
        self.solvent = solvent
        self.polymer = polymer
        # Constraints:
        #   phi in [0,1] 
        #   head,tail,thickness,interface>0
        #   head+tail >= thickness
    def parameters(self):
        return dict(solvent=self.solvent.parameters(),
                    polymer=self.polymer.parameters(),
                    thickness=self.thickness,
                    interface=self.interface,
                    head=self.head,
                    tail=self.tail,
                    phi=self.phi,
                    Y = self.Y)
    def render(self, probe, slabs):
        thickness, sigma, phi, head, tail, Y \
            = [p.value for p in self.thickness, self.interface, 
               self.phi, self.head, self.tail, self.Y]
        phi /= 100. # % to fraction
        Mr,Mi = self.polymer.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try: M,S = M[0],S[0]  # Temporary hack
        except: pass
        H = M*phi + S*(1-phi)
        head_width = head if head < thickness else thickness
        tail_width = tail if head+tail < thickness else thickness-head_width
        solvent_width = thickness - (head_width+tail_width)
        if solvent_width < 0: solvent_width = 0

        Pw,Pz = slabs.microslabs(tail_width)
        profile = phi * (1 - (Pz/tail)**2)**Y
        P = M*profile + S*(1-profile)
        #P.reshape((1,len(profile)))

        slabs.extend(rho = [real(H)], irho = [imag(H)], w = [head_width])
        slabs.extend(rho = [real(P)], irho = [imag(P)], w = Pw)
        slabs.extend(rho = [real(S)], irho = [imag(S)], w = [solvent_width],
                     sigma=[sigma])

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
