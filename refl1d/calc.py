#!/usr/bin/env python

"""
Basic reflectometry calculations.

reflectivity, reflectivity_amplitude:
    Slab model with optional absorption and roughness.  The function
    reflectivity returns the magnitude squared of the waveform.  The
    function reflectivity_amplitude returns the complex waveform.

magnetic_reflectivity, magnetic_amplitude, unpolarized_magnetic:
    Slab model with supporting magnetic scattering.  The function
    magnetic_reflectivity returns the magnitude squared for the four
    spin polarization cross sections [++, +-, -+, --].  The function
    magnetic_amplitude returns the complex waveforms. The function
    unpolarized_magnetic returns the expected magnitude for a measurement
    of the magnetic scattering using an unpolarized beam.

convolve, fixedres, varyingres
    Functions for estimating the resolution and convolving the
    the profile with the resolution function.
"""

__doc__ = "Fundamental reflectivity calculations"
__author__ = "Paul Kienzle"
__all__ = [ 'reflectivity', 'reflectivity_amplitude',
            'magnetic_reflectivity', 'magnetic_amplitude',
            'unpolarized_magnetic',
            'fixedres', 'varyingres', 'convolve'
            ]

import numpy
from numpy import pi, sin, cos, conj
from numpy import ascontiguousarray as _dense
from . import reflmodule


def reflectivity(*args, **kw):
    """
    Return reflectivity R^2 from slab model with sharp interfaces.
    returns reflectivities.

    The parameters are as follows:

    Q (angstrom**-1)
        points at which to evaluate the reflectivity
    depth (angstrom)
        thickness of the individual layers (incident and substrate
        depths are ignored)
    rho (microNb)
        Scattering length density.
    mu (microNb)
        absorption. Defaults to 0.
    sigma (angstrom)
        interface roughness between the current layer and the next.
        The final layer is ignored.  This may be a scalar for fixed
        roughness on every layer, or None if there is no roughness.
    wavelength (angstrom)
        Incident wavelength (only affects absorption).  May be a vector.
        Defaults to 1.

    This function does not compute any instrument resolution corrections.

    Use reflectivity_amplitude to return the complex waveform.
    """
    r = reflectivity_amplitude(*args,**kw)
    return (r*conj(r)).real

def reflectivity_amplitude(Q,
                           depth,
                           rho,
                           mu=0,
                           sigma=None,
                           wavelength=1,
                           ):
    """
    Returns the complex reflectivity waveform.

    See reflectivity for details.
    """
    Q = _dense(Q,'d')
    R = numpy.empty(Q.shape,'D')

    n = len(depth)
    if numpy.isscalar(wavelength):
        wavelength=wavelength*numpy.ones(Q.shape, 'd')
    if numpy.isscalar(mu):
        mu = mu*numpy.ones(n, 'd')
    if numpy.isscalar(sigma):
        sigma = sigma*numpy.ones(n-1, 'd')

    wavelength,depth,rho,mu = [_dense(v,'d')
                                 for v in wavelength,depth,rho,mu]

    rho,mu = [v*1e-6 for v in rho,mu]
    if sigma is not None:
        sigma = _dense(sigma, 'd')
        reflmodule._reflectivity_amplitude_rough(rho, mu, depth, sigma, wavelength, Q, R)
    else:
        reflmodule._reflectivity_amplitude (rho, mu, depth, wavelength, Q, R)
    return R


def magnetic_reflectivity(*args,**kw):
    """
    Magnetic reflectivity for slab models.

    Returns the expected values for the four polarization cross
    sections (++,+-,-+,--).

    Return reflectivity R^2 from slab model with sharp interfaces.
    returns reflectivities.

    The parameters are as follows:

    Q (angstrom**-1)
        points at which to evaluate the reflectivity
    depth (angstrom)
        thickness of the individual layers (incident and substrate
        depths are ignored)
    rho (microNb)
        Scattering length density.
    mu (microNb)
        absorption. Defaults to 0.
    wavelength (angstrom)
        Incident wavelength (only affects absorption).  May be a vector.
        Defaults to 1.
    rho_m (microNb)
        Magnetic scattering length density correction.
    theta_m (degrees)
        Angle of the magnetism within the layer.
    Aguide (degrees)
        Angle of the guide field; -90 is the usual case

    This function does not compute any instrument resolution corrections
    or interface diffusion

    Use magnetic_amplitude to return the complex waveform.
    """
    r = magnetic_amplitude(*args,**kw)
    return [(z*z.conj()).real for z in r]

def unpolarized_magnetic(*args,**kw):
    """
    Returns the average of magnetic reflectivity for all cross-sections.

    See magnetic_reflectivity for details.
    """
    return reduce(numpy.add, magnetic_reflectivity(*args,**kw))/2.

def magnetic_amplitude(Q,
                       depth,
                       rho,
                       mu=0,
                       wavelength=1,
                       rho_m=0,
                       theta_m=0,
                       Aguide=-90.0
                       ):
    """
    Returns the complex magnetic reflectivity waveform.

    See magnetic_reflectivity for details.
    """
    Q = _dense(Q,'d')
    n = len(depth)
    if numpy.isscalar(wavelength):
        wavelength=wavelength*numpy.ones(Q.shape, 'd')
    if numpy.isscalar(mu):
        mu = mu*numpy.ones(n, 'd')
    if numpy.isscalar(rho_m):
        rho_m = rho_m*numpy.ones(n, 'd')
    if numpy.isscalar(theta_m):
        theta_m = theta_m*numpy.ones(n, 'd')

    depth,rho,mu,rho_m,wavelength,theta_m \
        = [_dense(a,'d') for a in depth, rho, mu,
           rho_m, wavelength, theta_m]
    R1,R2,R3,R4 = [numpy.empty(Q.shape,'D') for pol in 1,2,3,4]
    expth = cos(theta_m * pi/180.0) + 1j*sin(theta_m * pi/180.0)

    rho,mu,rho_m = [v*1e-6 for v in rho,mu,rho_m]
    reflmodule._magnetic_amplitude(rho, mu, depth, wavelength,
                                   rho_m,  expth, Aguide, Q,
                                   R1, R2, R3, R4
                                   )
    return R1,R2,R3,R4


def fixedres(wavelength,dLoL,dT,Q):
    """
    Return resolution dQ for fixed slits.


    Angular divergence dT is (s1+s2)/d, where d is the distance
    between the slits and s1,s2 is the slit openings.  Slits and
    distances should use the same units.
    """
    dQ = numpy.empty(Q.shape,'d')
    reflmodule._fixedres(wavelength,dLoL,dT,_dense(Q,'d'),dQ)
    return dQ


def varyingres(wavelength,dLoL,dToT,Q):
    """
    Return resolution dQ for varying slits.

    Angular divergence dT/T is (s1+s2)/d/theta(Q(s1,s2)).
    """
    dQ = numpy.empty(Q.shape,'d')
    reflmodule._varyingres(wavelength,dLoL,dToT,_dense(Q,'d'),dQ)
    return dQ


def convolve(Qi,Ri,Q,dQ):
    """
    Return convolution R[k] of width dQ[k] at points Q[k].
    """
    R = numpy.empty(Q.shape,'d')
    reflmodule._convolve(_dense(Qi,'d'),_dense(Ri,'d'),
                         _dense(Q,'d'),_dense(dQ,'d'),R)
    return R
