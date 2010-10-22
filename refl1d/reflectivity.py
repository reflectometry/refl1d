#!/usr/bin/env python

"""
.. sidebar:: On this Page
    
        * :class:`Reflectivity <refl1d.reflectivity.reflectivity>`
        * :class:`Reflectivity amplitude <refl1d.reflectivity.reflectivity_amplitude>`
        * :class:`Magenetic reflectivity <refl1d.reflectivity.magnetic_reflectivity>`
     
Basic reflectometry calculations.

:class:`reflectivity <refl1d.reflectivity.reflectivity>`,
:class:`magnetic_reflectivity <refl1d.reflectivity.magnetic_reflectivity>`,
:class:`unpolarized_magnetic <refl1d.reflectivity.unpolarized_magnetic>`:
    Slab model reflectivity calculator with optional absorption and roughness.
    The function reflectivity_amplitude returns the complex waveform.
    Slab model with supporting magnetic scattering.  The function
    magnetic_reflectivity returns the complex reflection for the four
    spin polarization cross sections [++, +-, -+, --].  The function
    unpolarized_magnetic returns the expected magnitude for a measurement
    of the magnetic scattering using an unpolarized beam.

:class:`convolve <refl1d.reflectivity.convolve>`
    Apply Q-dependent resolution function to the theory.
"""

#__doc__ = "Fundamental reflectivity calculations"
__author__ = "Paul Kienzle"
__all__ = [ 'reflectivity', 'reflectivity_amplitude',
            'magnetic_reflectivity', 'magnetic_amplitude',
            'unpolarized_magnetic', 'convolve',
            'erf'
            ]

import numpy
from numpy import pi, sin, cos, conj
from numpy import ascontiguousarray as _dense
from . import reflmodule


def reflectivity(*args, **kw):
    """
    Calculate reflectivity |r(k_z)|^2 from slab model.

    :Parameters :
        *depth* : float[N] | angstrom
            Thickness of the individual layers (incident and substrate
            depths are ignored)
        *sigma* : float OR float[N-1] | angstrom
            Interface roughness between the current layer and the next.
            The final layer is ignored.  This may be a scalar for fixed
            roughness on every layer, or None if there is no roughness.
        *rho*, *irho* : float[N] OR float[N,K] | 10^-6 angstrom**-2
            Real and imaginary scattering length density.  Use multiple
            columns when you have kz-dependent scattering length densities,
            and set rho_offset to select the appropriate one.  Data should
            be stored in column order.
        *kz* : float[M] | angstrom**-1
            Points at which to evaluate the reflectivity
        *rho_index* : integer[M]
            *rho* and *irho* columns to use for the various kz.

    :Returns:
        *R* | float[M]
            Reflectivity magnitude.

    This function does not compute any instrument resolution corrections.
    """
    r = reflectivity_amplitude(*args,**kw)
    return (r*conj(r)).real

def reflectivity_amplitude(kz=None,
                           depth=None,
                           rho=None,
                           irho=0,
                           sigma=0,
                           rho_index=None,
                           ):
    """
    Calculate reflectivity r(k_z) from slab model.

    :Parameters :
        *depth* : float[N] | angstrom
            Thickness of the individual layers (incident and substrate
            depths are ignored)
        *sigma* = 0 : float OR float[N-1] | angstrom
            Interface roughness between the current layer and the next.
            The final layer is ignored.  This may be a scalar for fixed
            roughness on every layer, or None if there is no roughness.
        *rho*, *irho* = 0: float[N] OR float[N,K] | 10^-6 angstrom**-2
            Real and imaginary scattering length density.  Use multiple
            columns when you have kz-dependent scattering length densities,
            and set *rho_index* to select amongst them.  Data should be
            stored in column order.
        *kz* : float[M] | angstrom**-1
            Points at which to evaluate the reflectivity
        *rho_index* = 0 : integer[M]
            *rho* and *irho* columns to use for the various kz.

    :Returns:
        *r* | complex[M]
            Complex reflectivity waveform.

    This function does not compute any instrument resolution corrections.
    """
    kz = _dense(kz, 'd')
    if rho_index == None:
        rho_index = numpy.zeros(kz.shape,'i')
    else:
        rho_index = _dense(rho_index, 'i')

    depth = _dense(depth, 'd')
    if numpy.isscalar(sigma):
        sigma = sigma*numpy.ones(len(depth)-1, 'd')
    else:
        sigma = _dense(sigma, 'd')
    rho = _dense(rho, 'd')
    if numpy.isscalar(irho):
        irho = irho * numpy.ones_like(rho)
    else:
        irho = _dense(irho, 'd')

    r = numpy.empty(kz.shape,'D')
    reflmodule._reflectivity_amplitude(depth, sigma, rho, irho, kz, rho_index, r)
    return r


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

    See :class:`magnetic_reflectivity <refl1d.reflectivity.magnetic_reflectivity>` for details.
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

    See :class:`magnetic_reflectivity <refl1d.reflectivity.magnetic_reflectivity>` for details.
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

def convolve(Qi,Ri,Q,dQ):
    """
    Return convolution R[k] of width dQ[k] at points Q[k].
    """
    R = numpy.empty(Q.shape,'d')
    reflmodule._convolve(_dense(Qi),_dense(Ri),_dense(Q),_dense(dQ),R)
    return R

def erf(x):
    input = _dense(x,'d')
    output = numpy.empty_like(input)
    reflmodule._erf(input,output)
    return output
