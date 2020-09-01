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
from __future__ import print_function

from six.moves import reduce

#__doc__ = "Fundamental reflectivity calculations"
__author__ = "Paul Kienzle"
__all__ = ['reflectivity', 'reflectivity_amplitude',
           'magnetic_reflectivity', 'magnetic_amplitude',
           'unpolarized_magnetic', 'convolve',
          ]

import numpy as np
from numpy import pi, sin, cos, conj, radians
# delay load so doc build doesn't require compilation
#from . import reflmodule

BASE_GUIDE_ANGLE = 270.0

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
        *rho*, *irho* : float[N] OR float[N, K] | |1e-6/Ang^2|
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
    r = reflectivity_amplitude(*args, **kw)
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
        *rho*, *irho* = 0: float[N] OR float[N, K] | |1e-6/Ang^2|
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
        rho_index = np.zeros(kz.shape, 'i')
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


    #print(irho.shape, irho[:,0], irho[:,-1])
    irho[irho < 0] = 0.
    #print depth.shape, rho.shape, irho.shape, sigma.shape
    #print depth.dtype, rho.dtype, irho.dtype, sigma.dtype
    r = np.empty(kz.shape, 'D')
    #print "amplitude", depth, rho, kz, rho_index
    #print depth.shape, sigma.shape, rho.shape, irho.shape, kz.shape
    reflmodule._reflectivity_amplitude(depth, sigma, rho, irho, kz,
                                       rho_index, r)
    return r


def magnetic_reflectivity(*args, **kw):
    """
    Magnetic reflectivity for slab models.

    Returns the expected values for the four polarization cross
    sections (++, +-, -+, --).
    Return reflectivity R^2 from slab model with sharp interfaces.
    returns reflectivities.

    The parameters are as follows:

    kz (|1/Ang|)
        points at which to evaluate the reflectivity
    depth (|Ang|)
        thickness of the individual layers (incident and substrate
        depths are ignored)
    rho (|1e-6/Ang^2|)
        Scattering length density.
    irho (|1e-6/Ang^2|)
        absorption. Defaults to 0.
    rho_m (microNb)
        Magnetic scattering length density correction.
    theta_m (degrees)
        Angle of the magnetism within the layer.
    sigma (|Ang|)
        Interface roughness between the current layer and the next.
        The final layer is ignored.  This may be a scalar for fixed
        roughness on every layer, or None if there is no roughness.
    wavelength (|Ang|)
        Incident wavelength (only affects absorption).  May be a vector.
        Defaults to 1.
    Aguide (degrees)
        Angle of the guide field; -90 is the usual case

    This function does not compute any instrument resolution corrections.
    Interface diffusion, if present, uses the Nevot-Croce approximation.

    Use magnetic_amplitude to return the complex waveform.
    """
    r = magnetic_amplitude(*args, **kw)
    return [(z*z.conj()).real for z in r]

def unpolarized_magnetic(*args, **kw):
    """
    Returns the average of magnetic reflectivity for all cross-sections.

    See :class:`magnetic_reflectivity <refl1d.reflectivity.magnetic_reflectivity>` for details.
    """
    return reduce(np.add, magnetic_reflectivity(*args, **kw))/2.

B2SLD = 2.31604654  # Scattering factor for B field 1e-6/

def magnetic_amplitude(kz,
                       depth,
                       rho,
                       irho=0,
                       rhoM=0,
                       thetaM=0,
                       sigma=0,
                       Aguide=-90,
                       H=0,
                       rho_index=None,
                      ):
    """
    Returns the complex magnetic reflectivity waveform.

    See :class:`magnetic_reflectivity <refl1d.reflectivity.magnetic_reflectivity>` for details.
    """
    from . import reflmodule

    kz = _dense(kz, 'd')
    if rho_index is None:
        rho_index = np.zeros(kz.shape, 'i')
    else:
        rho_index = _dense(rho_index, 'i')
    n = len(depth)
    if np.isscalar(irho):
        irho = irho*np.ones(n, 'd')
    if np.isscalar(rhoM):
        rhoM = rhoM*np.ones(n, 'd')
    if np.isscalar(thetaM):
        thetaM = thetaM*np.ones(n, 'd')
    if np.isscalar(sigma):
        sigma = sigma*np.ones(n-1, 'd')

    #kz = -kz
    #depth, rho, irho, sigma, rhoM, thetaM = [v[::-1] for v in (depth, rho, irho, sigma, rhoM, thetaM)]
    depth, rho, irho, sigma = [_dense(a, 'd') for a in (depth, rho, irho, sigma)]
    #np.set_printoptions(linewidth=1000)
    #print(np.vstack((depth, np.hstack((sigma, np.nan)), rho, irho, rhoM, thetaM)).T)

    sld_b, u1, u3 = calculate_u1_u3(H, rhoM, thetaM, Aguide)

    R1, R2, R3, R4 = [np.empty(kz.shape, 'D') for pol in (1, 2, 3, 4)]
    reflmodule._magnetic_amplitude(depth, sigma, rho, irho,
                                   sld_b, u1, u3, Aguide, kz, rho_index,
                                   R1, R2, R3, R4)
    return R1, R2, R3, R4

def calculate_u1_u3(H, rhoM, thetaM, Aguide):
    from . import reflmodule

    rhoM, thetaM = _dense(rhoM, 'd'), _dense(np.radians(thetaM), 'd')
    n = len(rhoM)
    u1, u3 = np.empty(n, 'D'), np.empty(n, 'D')
    sld_b = np.empty(n, 'd')
    reflmodule._calculate_u1_u3(H, rhoM, thetaM, Aguide, sld_b, u1, u3)

    return sld_b, u1, u3

def calculate_u1_u3_py(H, rhoM, thetaM, Aguide):
    rotate_M = True

    EPS = np.finfo('f').tiny # not 1e-20 # epsilon offset for divisions.
    thetaM = radians(thetaM)
    phiH = radians(Aguide - BASE_GUIDE_ANGLE)
    thetaH = np.pi/2.0 # by convention, H is in y-z plane so theta = pi/2

    sld_h = B2SLD * H
    sld_m_x = rhoM * np.cos(thetaM)
    sld_m_y = rhoM * np.sin(thetaM)
    sld_m_z = 0.0 # by Maxwell's equations, H_demag = mz so we'll just cancel it here
    # The purpose of AGUIDE is to rotate the z-axis of the sample coordinate
    # system so that it is aligned with the quantization axis z, defined to be
    # the direction of the magnetic field outside the sample.
    if rotate_M:
        # rotate the M vector instead of the transfer matrix!
        # First, rotate the M vector about the x axis:
        new_my = sld_m_z * sin(radians(Aguide)) + sld_m_y * cos(radians(Aguide))
        new_mz = sld_m_z * cos(radians(Aguide)) - sld_m_y * sin(radians(Aguide))
        sld_m_y, sld_m_z = new_my, new_mz
        sld_h_x = sld_h_y = 0.0
        sld_h_z = sld_h
        # Then, don't rotate the transfer matrix
        Aguide = 0.0
    else:
        sld_h_x = sld_h * np.cos(thetaH) # zero
        sld_h_y = sld_h * np.sin(thetaH) * np.cos(phiH)
        sld_h_z = sld_h * np.sin(thetaH) * np.sin(phiH)

    sld_b_x = sld_h_x + sld_m_x
    sld_b_y = sld_h_y + sld_m_y
    sld_b_z = sld_h_z + sld_m_z

    # avoid divide-by-zero:
    sld_b_x += EPS * (sld_b_x == 0)
    sld_b_y += EPS * (sld_b_y == 0)

    # add epsilon to y, to avoid divide by zero errors?
    sld_b = np.sqrt(sld_b_x**2 + sld_b_y**2 + sld_b_z**2)
    u1_num = (+sld_b + sld_b_x + 1j*sld_b_y - sld_b_z)
    u1_den = (+sld_b + sld_b_x - 1j*sld_b_y + sld_b_z)
    u3_num = (-sld_b + sld_b_x + 1j*sld_b_y - sld_b_z)
    u3_den = (-sld_b + sld_b_x - 1j*sld_b_y + sld_b_z)

    u1 = u1_num/u1_den
    u3 = u3_num/u3_den
    #print "u1", u1
    #print "u3", u3
    return sld_b, u1, u3

try:
    #raise ImportError() # uncomment to force numba off
    from numba import njit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    # if no numba then njit does nothing
    def njit(*args, **kw):
        # Check for bare @njit, in which case we just return the function.
        if len(args) == 1 and callable(args[0]) and not kw:
            return args[0]
        # Otherwise we have @njit(...), so return the identity decorator.
        return lambda fn: fn

@njit('(f8[:], f8[:], f8[:], f8[:], f8[:])')
def _convolve_uniform(xi, yi, x, dx, y):
    root_12_over_2 = np.sqrt(3)
    left_index = 0
    N = len(xi)
    for k, x_k in enumerate(x):
        # Convert 1-sigma width to 1/2 width of the region
        limit = dx[k] * root_12_over_2
        #print(f"point {x_k} +/- {limit}")
        # Find integration limits, bound by the range of the data
        left, right = max(x_k - limit, xi[0]), min(x_k + limit, xi[-1])
        if right < left:
            # Convolution does not overlap data range.
            y[k] = 0.
            continue

        # Find the starting point for the convolution by first scanning
        # forward until we reach the next point greater than the limit
        # (we might already be there if the next output point has wider
        # resolution than the current point), then scanning backwards to
        # get to the last point before the limit. Make sure we have at
        # least one interval so that we don't have to check edge cases
        # later.
        while left_index < N-2 and xi[left_index] < left:
            left_index += 1
        while left_index > 0 and xi[left_index] > left:
            left_index -= 1

        # Set the first interval.
        total = 0.
        right_index = left_index + 1
        x1, y1 = xi[left_index], yi[left_index]
        x2, y2 = xi[right_index], yi[right_index]


        # Subtract the excess from left interval before the left edge.
        #print(f" left {left} in {(x1, y1)}, {(x2, y2)}")
        if x1 < left:
            # Subtract the area of the rectangle from (x1, 0) to (left, y1)
            # plus 1/2 the rectangle from (x1, y1) to (left, y'),
            # where y' is y value where the line (x1, y1) to (x2, y2)
            # intersects x=left. This can be computed as follows:
            #    offset = left - x1
            #    slope = (y2 - y1)/(x2 - x1)
            #    yleft = y1 + slope*offset
            #    area = offset * y1 + offset * (yleft-y1)/2
            # It can be simplified to the following:
            #    area = offset * (y1 + slope*offset/2)
            offset = left - x1
            slope = (y2 - y1)/(x2 - x1)
            area = offset * (y1 + 0.5*slope*offset)
            total -= area
            #print(f" left correction {area}")

        # Do trapezoidal integration up to and including the end interval
        while right_index < N-1 and x2 < right:
            # Add the current interval if it isn't empty
            if x1 != x2:
                area = 0.5*(y1 + y2)*(x2 - x1)
                total += area
                #print(f" adding {(x1,y1)}, {(x2, y2)} as {area}")
            # Move to the next interval
            right_index += 1
            x1, y1, x2, y2 = x2, y2, xi[right_index], yi[right_index]
        if x1 != x2:
            area = 0.5*(y1 + y2)*(x2 - x1)
            total += area
            #print(f" adding final {(x1,y1)}, {(x2, y2)} as {area}")

        # Subtract the excess from the right interval after the right edge.
        #print(f" right {right} in {(x1, y1)}, {(x2, y2)}")
        if x2 > right:
            # Expression for area to subtract using rectangles is as follows:
            #    offset = x2 - right
            #    slope = (y2 - y1)/(x2 - x1)
            #    yright = y2 - slope*offset
            #    area = -(offset * yright + offset * (y2-yright)/2)
            # It can be simplified to the following:
            #    area = -offset * (y2 - slope*offset/2)
            offset = x2 - right
            slope = (y2 - y1)/(x2 - x1)
            area = offset * (y2 - 0.5*slope*offset)
            total -= area
            #print(f" right correction {area}")

        # Normalize by interval length
        if left < right:
            #print(f" normalize by length {right} - {left}")
            y[k] = total / (right - left)
        elif x1 < x2:
            # If dx = 0 using the value interpolated at x (with left=right=x).
            #print(f" dirac delta at {left} = {right} in {(x1, y1)}, {(x2, y2)}")
            offset = left - x1
            slope = (y2 - y1)/(x2 - x1)
            y[k] = y1 + slope*offset
        else:
            # At an empty interval in the theory function. Average the y.
            #print(f" empty interval with {left} = {right} in {(x1, y1)}, {(x2, y2)}")
            y[k] = 0.5*(y1 + y2)

def convolve(xi, yi, x, dx, resolution='normal'):
    r"""
    Apply x-dependent gaussian resolution to the theory.

    Returns convolution y[k] of width dx[k] at points x[k].

    The theory function is a piece-wise linear spline which does not need to
    be uniformly sampled.  The theory calculation points *xi* should be dense
    enough to capture the "wiggle" in the theory function, and should extend
    beyond the ends of the data measurement points *x*. Convolution at the
    tails is truncated and normalized to area of overlap between the resolution
    function in case the theory does not extend far enough.

    *resolution* is 'normal' (default) or 'uniform'. Note that the uniform
    distribution uses the $1-\sigma$ equivalent distribution width which is
    $1/\sqrt{3}$ times the width of the rectangle.
    """
    from . import reflmodule

    xi, yi, x, dx = _dense(xi), _dense(yi), _dense(x), _dense(dx)
    y = np.empty_like(x)
    if resolution == 'uniform':
        _convolve_uniform(xi, yi, x, dx, y)
    else:
        reflmodule.convolve(xi, yi, x, dx, y)
    return y


def convolve_sampled(xi, yi, xp, yp, x, dx):
    """
    Apply x-dependent arbitrary resolution function to the theory.

    Returns convolution y[k] of width dx[k] at points x[k].

    Like :func:`convolve`, the theory *(xi, yi)* is represented as a
    piece-wise linear spline which should extend beyond the data
    measurement points *x*.  Instead of a gaussian resolution function,
    resolution *(xp, yp)* is also represented as a piece-wise linear
    spline.
    """
    from . import reflmodule

    x = _dense(x)
    y = np.empty_like(x)
    reflmodule.convolve_sampled(_dense(xi), _dense(yi), _dense(xp), _dense(yp),
                                x, _dense(dx), y)
    return y

def test_uniform():
    xi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yi = [1, 3, 1, 2, 7, 3, 1, 2, 1, 3]
    _check_uniform("uniform aligned", xi, yi, x=[2, 4, 6, 8], dx=1)
    _check_uniform("uniform unaligned", xi, yi, x=[2.5, 4.5, 6.5, 8.5], dx=1)
    _check_uniform("uniform wide", xi, yi, x=[2.5, 4.5, 6.5, 8.5], dx=3)
    # Check bad values
    ystar = convolve(xi, yi, x=[-3, 13], dx=[1/np.sqrt(3)]*2, resolution='uniform')
    assert (ystar == 0.).all()

    xi = [1.1, 2.3, 2.8, 4.2, 5, 6, 7, 8, 9, 10]
    yi = [1, 3, 1, 2, 7, 3, 1, 2, 1, 3]
    _check_uniform("uniform unaligned", xi, yi, x=[2.5, 4.5, 6.5, 8.5], dx=1)

def _check_uniform(name, xi, yi, x, dx):
    # Note: using fixed dx since that's all _check_spline supports.
    ystar = convolve(xi, yi, x, [dx/np.sqrt(3)]*len(x), resolution='uniform')
    #print("xi", xi)
    #print("yi", yi)
    #print("x", x, "dx", dx)
    #print("ystar", ystar)
    xp = [-dx, -dx, dx, dx]
    yp = [0, 0.5/dx, 0.5/dx, 0]
    _check_spline(name, xi, yi, xp, yp, x, ystar)

def test_convolve_sampled():
    xi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yi = [1, 3, 1, 2, 1, 3, 1, 2, 1, 3]
    xp = [-1, 0, 1, 2, 3]
    yp = [1, 4, 3, 2, 1]
    _check_sampled("sampled aligned", xi, yi, xp, yp, dx=1)
    _check_sampled("sampled unaligned", xi, yi, _dense(xp) - 0.2000003, yp, dx=1)
    _check_sampled("sampled wide", xi, yi, xp, yp, dx=2)
    _check_sampled("sampled super wide", xi, yi, xp, yp, dx=10)

def _check_sampled(name, xi, yi, xp, yp, dx):
    ystar = convolve_sampled(xi, yi, xp, yp, xi, dx=np.full_like(xi, dx))
    xp = np.array(xp)*dx
    _check_spline(name, xi, yi, xp, yp, xi, ystar)

def _check_spline(name, xi, yi, xp, yp, x, ystar):
    step = 0.0001
    xpfine = np.arange(xp[0], xp[-1] + step / 10, step)
    ypfine = np.interp(xpfine, xp, yp)
    # make sure xfine is wide enough by adding an extra interval at the ends
    xfine = np.arange(xi[0] + xpfine[0] - 2*step, xi[-1] + xpfine[-1] + 2*step, step)
    yfine = np.interp(xfine, xi, yi, left=0, right=0)
    pidx = np.searchsorted(xfine, np.array(x) + xp[0])
    left, right = np.searchsorted(xfine, [xi[0], xi[-1]])

    conv = []
    for pk in pidx:
        norm_start = max(0, left - pk)
        norm_end = min(len(xpfine), right - pk)
        norm = step * np.sum(ypfine[norm_start:norm_end])
        conv.append(step * np.sum(ypfine * yfine[pk:pk + len(xpfine)]) / norm)

    #print("checking convolution %s"%(name, ))
    #print(" ".join("%7.4f"%yi for yi in ystar))
    #print(" ".join("%7.4f"%yi for yi in conv))
    assert all(abs(yi - fi) < 0.0005 for (yi, fi) in zip(ystar, conv)), name
