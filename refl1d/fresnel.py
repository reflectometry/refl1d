# This code is public domain

"""
Pure python Fresnel reflectivity calculator.
"""

from numpy import sqrt, exp, real, conj, pi, log, abs, choose

class Fresnel:
    """
    Function for computing the Fresnel reflectivity for a single interface.

    :Parameters:
        *rho*, *irho* = 0 : float | 1e6 * inv Angstrom^2
            real and imaginary scattering length density of backing medium
        *Vrho*, *Virho* = 0 : float | 1e6 * inv Angstrom^2
            real and imaginary scattering length density of incident medium
        *sigma* = 0 : float | Angstrom
            interfacial roughness
            
    :Returns:
        fresnel : Fresnel
            callable object for computing Fresnel reflectivity at Q

    Note that we do not correct for attenuation of the beam through the 
    incident medium since we do not know the path length.
    """
    def __init__(self, rho=0, irho=0, sigma=0, Vrho=0, Virho=0):
        self.rho,self.Vrho,self.irho,self.Virho,self.sigma \
            = rho,Vrho,irho,Virho,sigma

    def reflectivity(self, Q):
        """
        Compute the Fresnel reflectivity at the given Q/wavelength.
        """
        # If Q < 0, then we are going from substrate into incident medium.
        # In that case we must negate the change in scattering length density
        # and ignore the absorption.
        Qp_rho = 1e-6*(self.rho-self.Vrho + 1j*self.irho)
        Qm_rho = 1e-6*(self.Vrho-self.rho + 1j*self.Virho)
        rho = choose(Q<0, (Qm_rho,Qp_rho))
        kz = abs(Q)/2
        f = sqrt(kz**2 - 4*pi*rho)  # fresnel coefficient

        # Compute reflectivity amplitude, with adjustment for roughness
        amp = (kz-f)/(kz+f) * exp(-2*self.sigma**2*kz*f)
        # Note: we do not need to check for a divide by zero.
        # Qc^2 = 16 pi rho.  Since rho is non-zero then Qc is non-zero.
        # For mu = 0:
        # * If |Qz| < Qc then f has an imaginary component, so |Qz|+f != 0.
        # * If |Qz| > Qc then |Qz| > 0 and f > 0, so |Qz|+f != 0.
        # * If |Qz| = Qc then |Qz| != 0 and f = 0, so |Qz|+f != 0.
        # For mu != 0:
        # * f has an imaginary component, so |Q|+f != 0.

        R = real(amp*conj(amp))
        return R

    # Make the reflectivity method the default
    __call__ = reflectivity

def test():
    import numpy
    import abeles
    
    # Rough silicon with an anomolously large absorbtion
    rho,irho = 2.07,0.01
    Vrho,Virho = -1,0.1
    sigma = 20
    fresnel = Fresnel(rho=rho, irho=irho, Vrho=Vrho, Virho=Virho, sigma=sigma)

    Mw = [0,0]
    Mrho = [[rho,Vrho]]
    Mirho = [[irho,Virho]]
    Msigma = [sigma]

    Q = numpy.linspace(-0.1,0.1,11,'d')
    Rf = fresnel(Q)
    rm = abeles.refl(Q/2, depth=Mw, rho=Mrho, irho=Mirho, sigma=Msigma)
    Rm = abs(rm)**2

    relerr = numpy.linalg.norm((Rf-Rm)/Rm)
    assert relerr < 1e-10, "relative error is %g"%relerr

if __name__ == "__main__": test()
