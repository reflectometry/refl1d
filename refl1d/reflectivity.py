"""
Basic reflectometry calculations

Slab model reflectivity calculator with optional absorption and roughness.
The function reflectivity_amplitude returns the complex waveform.
Slab model with supporting magnetic scattering.  The function
magnetic_reflectivity returns the complex reflection for the four
spin polarization cross sections [++, +-, -+, --].  The function
unpolarized_magnetic returns the expected magnitude for a measurement
of the magnetic scattering using an unpolarized beam.
"""
from six.moves import reduce

#__doc__ = "Fundamental reflectivity calculations"
__author__ = "Paul Kienzle"
__all__ = [ 'reflectivity', 'reflectivity_amplitude',
            'magnetic_reflectivity', 'magnetic_amplitude',
            'unpolarized_magnetic', 'convolve',
            ]

import numpy as np
from numpy import pi, sin, cos, conj
# delay load so doc build doesn't require compilation
#from . import reflmodule

def _dense(x, dtype='d'):
    return np.ascontiguousarray(x, dtype)


def reflectivity(*args, **kw):
    """
    Calculate reflectivity $|r(k_z)|^2$ from slab model.

    :Parameters :
        *depth* : float[N] | |Ang|
            Thickness of the individual layers (incident and substrate
            depths are ignored)
        *sigma* : float OR float[N-1] | |Ang|
            Interface roughness between the current layer and the next.
            The final layer is ignored.  This may be a scalar for fixed
            roughness on every layer, or None if there is no roughness.
        *rho*, *irho* : float[N] OR float[N,K] | |1e-6/Ang^2|
            Real and imaginary scattering length density.  Use multiple
            columns when you have kz-dependent scattering length densities,
            and set rho_offset to select the appropriate one.  Data should
            be stored in column order.
        *kz* : float[M] | |1/Ang|
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
    r"""
    Calculate reflectivity amplitude $r(k_z)$ from slab model.

    :Parameters :
        *depth* : float[N] | |Ang|
            Thickness of the individual layers (incident and substrate
            depths are ignored)
        *sigma* = 0 : float OR float[N-1] | |Ang|
            Interface roughness between the current layer and the next.
            The final layer is ignored.  This may be a scalar for fixed
            roughness on every layer, or None if there is no roughness.
        *rho*, *irho* = 0: float[N] OR float[N,K] | |1e-6/Ang^2|
            Real and imaginary scattering length density.  Use multiple
            columns when you have kz-dependent scattering length densities,
            and set *rho_index* to select amongst them.  Data should be
            stored in column order.
        *kz* : float[M] | |1/Ang|
            Points at which to evaluate the reflectivity
        *rho_index* = 0 : integer[M]
            *rho* and *irho* columns to use for the various kz.

    :Returns:
        *r* | complex[M]
            Complex reflectivity waveform.

    This function does not compute any instrument resolution corrections.
    """
    from . import reflmodule

    kz = _dense(kz, 'd')
    if rho_index is None:
        rho_index = np.zeros(kz.shape,'i')
    else:
        rho_index = _dense(rho_index, 'i')

    depth = _dense(depth, 'd')
    if np.isscalar(sigma):
        sigma = sigma*np.ones(len(depth)-1, 'd')
    else:
        sigma = _dense(sigma, 'd')
    rho = _dense(rho, 'd')
    if np.isscalar(irho):
        irho = irho * np.ones_like(rho)
    else:
        irho = _dense(irho, 'd')

    #print depth.shape,rho.shape,irho.shape,sigma.shape
    #print depth.dtype,rho.dtype,irho.dtype,sigma.dtype
    r = np.empty(kz.shape,'D')
    #print "amplitude",depth,rho,kz,rho_index
    #print depth.shape, sigma.shape, rho.shape, irho.shape, kz.shape
    reflmodule._reflectivity_amplitude(depth, sigma, rho, irho, kz,
                                       rho_index, r)
    return r


def magnetic_reflectivity(*args,**kw):
    """
    Magnetic reflectivity for slab models.

    Returns the expected values for the four polarization cross
    sections (++,+-,-+,--).
    Return reflectivity R^2 from slab model with sharp interfaces.
    returns reflectivities.

    The parameters are as follows:

    kz (|1/Ang|)
        points at which to evaluate the reflectivity
    depth (|Ang|)
        thickness of the individual layers (incident and substrate
        depths are ignored)
    rho (microNb)
        Scattering length density.
    mu (microNb)
        absorption. Defaults to 0.
    wavelength (|Ang|)
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
    return reduce(np.add, magnetic_reflectivity(*args,**kw))/2.

def magnetic_amplitude(kz,
                       depth,
                       rho,
                       irho=0,
                       rhoM=0,
                       thetaM=0,
                       phiM=0,
                       sigma=0,
                       Aguide=-90.0,
                       rho_index=None,
                       ):
    """
    Returns the complex magnetic reflectivity waveform.

    See :class:`magnetic_reflectivity <refl1d.reflectivity.magnetic_reflectivity>` for details.
    """
    from . import reflmodule

    kz = _dense(kz,'d')
    if rho_index is None:
        rho_index = np.zeros(kz.shape,'i')
    else:
        rho_index = _dense(rho_index, 'i')
    n = len(depth)
    if np.isscalar(irho):
        irho = irho*np.ones(n, 'd')
    if np.isscalar(rhoM):
        rhoM = rhoM*np.ones(n, 'd')
    if np.isscalar(thetaM):
        thetaM = thetaM*np.ones(n, 'd')
    if np.isscalar(phiM):
        phiM = phiM*np.ones(n, 'd')
    if np.isscalar(sigma):
        sigma = sigma*np.ones(n-1, 'd')

    depth, rho, irho, rhoM, thetaM, phiM, sigma \
        = [_dense(a,'d') for a in (depth, rho, irho, rhoM, thetaM, phiM, sigma)]
    thetaM *= pi/180.0
    phiM *= pi/180.0
    u1 = (1 + cos(thetaM) + 1j*sin(thetaM)*cos(phiM) - sin(thetaM)*sin(phiM)) / \
         (1 + cos(thetaM) - 1j*sin(thetaM)*cos(phiM) + sin(thetaM)*sin(phiM)) 
    u3 = (-1 + cos(thetaM) + 1j*sin(thetaM)*cos(phiM) - sin(thetaM)*sin(phiM)) / \
         (-1 + cos(thetaM) - 1j*sin(thetaM)*cos(phiM) + sin(thetaM)*sin(phiM)) 
    #rho,irho,rhoM = [v*1e-6 for v in rho,irho,rhoM]
    R1,R2,R3,R4 = [np.empty(kz.shape,'D') for pol in (1,2,3,4)]
    reflmodule._magnetic_amplitude(depth, sigma, rho, irho,
                                   rhoM, u1, u3, Aguide, kz, rho_index,
                                   R1, R2, R3, R4
                                   )
    return R1,R2,R3,R4


def convolve(xi, yi, x, dx):
    """
    Apply x-dependent gaussian resolution to the theory.

    Returns convolution y[k] of width dx[k] at points x[k].

    The theory function is a piece-wise linear spline which does not need to
    be uniformly sampled.  The theory calculation points *xi* should be dense
    enough to capture the "wiggle" in the theory function, and should extend
    beyond the ends of the data measurement points *x*. Convolution at the
    tails is truncated and normalized to area of overlap between the resolution
    function in case the theory does not extend far enough.
    """
    from . import reflmodule

    x = _dense(x)
    y = np.empty_like(x)
    reflmodule.convolve(_dense(xi), _dense(yi), x, _dense(dx), y)
    return y


def convolve_sampled(xi, yi, xp, yp, x, dx):
    """
    Apply x-dependent arbitrary resolution function to the theory.

    Returns convolution y[k] of width dx[k] at points x[k].

    Like :func:`convolve`, the theory *(xi,yi)* is represented as a
    piece-wise linear spline which should extend beyond the data
    measurement points *x*.  Instead of a gaussian resolution function,
    resolution *(xp,yp)* is also represented as a piece-wise linear
    spline.
    """
    from . import reflmodule

    x = _dense(x)
    y = np.empty_like(x)
    reflmodule.convolve_sampled(_dense(xi), _dense(yi), _dense(xp), _dense(yp),
                                x, _dense(dx), y)
    return y


def test_convolve_sampled():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 3, 1, 2, 1, 3, 1, 2, 1, 3]
    xp = [-1, 0, 1, 2, 3]
    yp = [1, 4, 3, 2, 1]
    _check_convolution("aligned", x, y, xp, yp, dx=1)
    _check_convolution("unaligned", x, y, _dense(xp) - 0.2000003, yp, dx=1)
    _check_convolution("wide", x, y, xp, yp, dx=2)
    _check_convolution("super wide", x, y, xp, yp, dx=10)


def _check_convolution(name, x, y, xp, yp, dx):
    ystar = convolve_sampled(x, y, xp, yp, x, dx=np.ones_like(x) * dx)

    xp = np.array(xp) * dx
    step = 0.0001
    xpfine = np.arange(xp[0], xp[-1] + step / 10, step)
    ypfine = np.interp(xpfine, xp, yp)
    # make sure xfine is wide enough by adding a couple of extra steps
    # at the end
    xfine = np.arange(x[0] + xpfine[0], x[-1] + xpfine[-1] + 2 * step, step)
    yfine = np.interp(xfine, x, y, left=0, right=0)
    pidx = np.searchsorted(xfine, np.array(x) + xp[0])
    left, right = np.searchsorted(xfine, [x[0], x[-1]])

    conv = []
    for pi in pidx:
        norm_start = max(0, left - pi)
        norm_end = min(len(xpfine), right - pi)
        norm = step * np.sum(ypfine[norm_start:norm_end])
        conv.append(
            step * np.sum(ypfine * yfine[pi:pi + len(xpfine)]) / norm)

    #print("checking convolution %s"%(name,))
    #print(" ".join("%7.4f"%yi for yi in ystar))
    #print(" ".join("%7.4f"%yi for yi in conv))
    assert all(abs(yi - fi) < 0.0005 for (yi, fi) in zip(ystar, conv))
