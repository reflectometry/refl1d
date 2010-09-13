"""
Two variants:

    direct: use the chebyshev polynomial coefficients directly
    interp: interpolate through control points to set the coefficients.

Control points are located at fixed z_k

    z_k = L * (cos( pi*(k-0.5)/N )+1)/2 for k in 1..N

where L is the thickness of the layer.
    
Interpolation as an O(N log N) cost to calculation of the profile for N
coefficients.  This is in addition to the O (N S) cost of the direct
profile for S microslabs.


"""
import numpy
from numpy import inf, real, imag, exp, pi
from numpy.fft import fft
from mystic import Parameter as Par, IntegerParameter as IntPar
from .model import Layer

#TODO: add left_sld,right_sld to all layers so that fresnel works
#TODO: access left_sld,right_sld so freeform doesn't need left,right
#TODO: restructure to use vector parameters
#TODO: allow the number of layers to be adjusted by the fit
class FreeformCheby(Layer):
    """
    A freeform section of the sample modeled with Chebyshev polynomials.

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    polynomial orders.
    """
    def __init__(self, thickness=0, rho=[], irho=[], 
                 name="Cheby", method="interp"):
        self.name = name
        self.method = method
        self.thickness = Par.default(thickness, limits=(0,inf),
                                   name=name+" thickness")
        self.rho,self.irho \
            = [[Par.default(p,name=name+"[%d] %s"%(i,part),limits=limits)
                for i,p in enumerate(v)]
               for v,part,limits in zip((rho, irho),
                                        ('rho', 'irho'),
                                        ((-inf,inf),(-inf,inf)),
                                        )]
    def parameters(self):
        return dict(rho=self.rho,
                    irho=self.irho,
                    thickness=self.thickness)
    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw,Pz = slabs.microslabs(thickness)
        t = Pz/thickness
        Prho = cheby_profile([p.value for p in self.rho], t, self.method)
        Pirho = cheby_profile([p.value for p in self.irho], t, self.method)        
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)

class ChebyVF(Layer):
    """
    Material in a solvent

    :Parameters:

        *thickness* : float | Angstrom
            the thickness of the solvent layer
        *interface* : float | Angstrom
            the rms roughness of the solvent surface
        *material* : Material
            the material of interest
        *solvent* : Material
            the solvent or vacuum
        *vf* : [float] 
            the control points for volume fraction
        *method* = 'interp' : string | 'direct' or 'interp'
            freeform profile method 

    *method* is 'direct' if the *vf* values refer to chebyshev
    polynomial coefficients or 'interp' if *vf* values refer to
    control points located at z_k

        z_k = L*(cos(pi (k-0.5)/N)+1) + 1)/2  for k=1..N.

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition 
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    These parameters combine in the following profile formula::
    
        sld(z) = material.sld * profile(z) + solvent.sld * (1 - profile(z))
    """
    def __init__(self, thickness=0, interface=0,
                 material=None, solvent=None,
                 length=None, vf=None, method="interp"):
        self.thickness = Par.default(thickness, name="solvent thickness")
        self.interface = Par.default(interface, name="solvent interface")
        self.length = Par.default(length, name="length")
        self.solvent = solvent
        self.material = material
        self.vf = [Par.default(p,name="vf[%d]"%i) for i,p in enumerate(vf)]
        self.method = method
        # Constraints:
        #   base_vf in [0,1] 
        #   base,length,sigma,thickness,interface>0
        #   base+length+3*sigma <= thickness
    def parameters(self):
        return dict(solvent=self.solvent.parameters(),
                    material=self.material.parameters(),
                    thickness=self.thickness,
                    interface=self.interface,
                    length=self.length,
                    vf=self.vf)
    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw,Pz = slabs.microslabs(thickness)
        t = Pz/thickness
        vf = cheby_profile([p.value for p in self.vf], t, self.method)
        vf = numpy.clip(vf,0,1)
        Mr,Mi = self.material.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try: M,S = M[0],S[0]  # Temporary hack
        except: pass
        P = M*vf + S*(1-vf)
        Pr, Pi = real(P), imag(P)        
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

def cheby_profile(control, t, method):
    n = len(control)
    if n == 0:
        return 0*t
    c = numpy.array(control)
    if method == 'interp':
        # DCT calculation of chebyshev coefficients
        #    c_j = 2/N sum_k=1^N f_k cos((2 pi j (k-1) / 2 N)
        # where 
        #    f_k = f[cos( (2 pi k - 1) / 2 N]
        w = exp((-0.5j*pi/n)*numpy.arange(n))
        y = numpy.hstack((c[0::2], c[1::2][::-1]))
        c = (2./n) * real(fft(y)*w)
    # Crenshaw recursion from numerical recipes sec. 5.8
    y = 4*t - 2
    d = dd = 0
    for c_j in c[:0:-1]:
        d, dd = y*d + (c_j - dd), d
    return y*(0.5*d) + (0.5*c[0] - dd)
