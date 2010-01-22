from numpy import pi, inf, sqrt, log, degrees, radians, cos, sin, tan
from numpy import arcsin as asin, ceil

_FWHM_scale = sqrt(log(256))
def FWHM2sigma(s):
    return s/_FWHM_scale
def sigma2FWHM(s):
    return s*_FWHM_scale

def QL2T(Q=None,L=None):
    """
    Compute angle from Q and wavelength.
    
    T = asin( |Q| L / 4 pi )
    
    Returns T in degrees.
    """
    return degrees(asin(abs(Q) * L / (4*pi)))

def TL2Q(T=None,L=None):
    """
    Compute Q from angle and wavelength.
    
    Q = 4 pi sin(T) / L
    
    Returns Q in inverse Angstroms.
    """
    return 4 * pi * sin(radians(T)) / L

def dTdL2dQ(T, dT, L, dL):
    """
    Convert wavelength dispersion and angular divergence to Q resolution.
    
    *T*,*dT*  (degrees) angle and FWHM angular divergence
    *L*,*dL*  (Angstroms) wavelength and FWHM wavelength dispersion

    Returns 1-sigma dQ
    """

    # Compute dQ from wavelength dispersion (dL) and angular divergence (dT)
    T,dT = radians(T), radians(dT)
    #print T, dT, L, dL
    dQ = (4*pi/L) * sqrt( (sin(T)*dL/L)**2 + (cos(T)*dT)**2 )

    return FWHM2sigma(dQ)

def dQdT2dLoL(Q, dQ, T, dT):
    """
    Convert a calculated Q resolution and wavelength divergence to a
    wavelength dispersion.
    
    *Q*, *dQ* (inv Angstroms)  Q and 1-sigma Q resolution
    *T*, *dT* (degrees) angle and FWHM angular divergence
    
    Returns FWHM dL/L
    """
    return sqrt( (sigma2FWHM(dQ)/Q)**2 - (radians(dT)/tan(radians(T)))**2 )

