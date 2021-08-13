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

# __doc__ = "Fundamental reflectivity calculations"
__author__ = "Paul Kienzle"
__all__ = ['reflectivity', 'reflectivity_amplitude',
           'magnetic_reflectivity', 'magnetic_amplitude',
           'unpolarized_magnetic', 'convolve',
          ]

import numpy as np
from numpy import pi, sin, cos, conj, radians, sqrt, exp, fabs
# delay load so doc build doesn't require compilation
# from . import reflmodule

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

    # print(irho.shape, irho[:,0], irho[:,-1])
    irho[irho < 0] = 0.
    # print depth.shape, rho.shape, irho.shape, sigma.shape
    # print depth.dtype, rho.dtype, irho.dtype, sigma.dtype
    r = np.empty(kz.shape, 'D')
    # print "amplitude", depth, rho, kz, rho_index
    # print depth.shape, sigma.shape, rho.shape, irho.shape, kz.shape
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

    # kz = -kz
    # depth, rho, irho, sigma, rhoM, thetaM = [v[::-1] for v in (depth, rho, irho, sigma, rhoM, thetaM)]
    depth, rho, irho, sigma = [_dense(a, 'd')
                                      for a in (depth, rho, irho, sigma)]
    # np.set_printoptions(linewidth=1000)
    # print(np.vstack((depth, np.hstack((sigma, np.nan)), rho, irho, rhoM, thetaM)).T)

    sld_b, u1, u3 = calculate_u1_u3(H, rhoM, thetaM, Aguide)

    # Note 2021-08-01: return Rpp, Rpm, Rmp, Rmm are no longer contiguous.
    R = np.empty((kz.size, 4), 'D')
    magnetic_amplitude_py(depth, sigma, rho, irho, sld_b, u1, u3, kz, R)
    return R[:, 0], R[:, 1], R[:, 2], R[:, 3]


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

    EPS = np.finfo('f').tiny  # not 1e-20 # epsilon offset for divisions.
    thetaM = radians(thetaM)
    phiH = radians(Aguide - BASE_GUIDE_ANGLE)
    thetaH = np.pi/2.0  # by convention, H is in y-z plane so theta = pi/2

    sld_h = B2SLD * H
    sld_m_x = rhoM * np.cos(thetaM)
    sld_m_y = rhoM * np.sin(thetaM)
    sld_m_z = 0.0  # by Maxwell's equations, H_demag = mz so we'll just cancel it here
    # The purpose of AGUIDE is to rotate the z-axis of the sample coordinate
    # system so that it is aligned with the quantization axis z, defined to be
    # the direction of the magnetic field outside the sample.
    if rotate_M:
        # rotate the M vector instead of the transfer matrix!
        # First, rotate the M vector about the x axis:
        new_my = sld_m_z * sin(radians(Aguide)) + \
                               sld_m_y * cos(radians(Aguide))
        new_mz = sld_m_z * cos(radians(Aguide)) - \
                               sld_m_y * sin(radians(Aguide))
        sld_m_y, sld_m_z = new_my, new_mz
        sld_h_x = sld_h_y = 0.0
        sld_h_z = sld_h
        # Then, don't rotate the transfer matrix
        Aguide = 0.0
    else:
        sld_h_x = sld_h * np.cos(thetaH)  # zero
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
    # print "u1", u1
    # print "u3", u3
    return sld_b, u1, u3


try:
    # raise ImportError() # uncomment to force numba off
    from numba import njit, prange
    import numba
    USE_NUMBA = True
    #numba.config.THREADING_LAYER = 'safe'
    #numba.set_num_threads(8)
except ImportError:
    USE_NUMBA = False
    # if no numba then njit does nothing

    def njit(*args, **kw):
        # Check for bare @njit, in which case we just return the function.
        if len(args) == 1 and callable(args[0]) and not kw:
            return args[0]
        # Otherwise we have @njit(...), so return the identity decorator.
        return lambda fn: fn

MINIMAL_RHO_M = 1e-2  # in units of 1e-6/A^2
EPS = np.finfo(float).eps
B2SLD = 2.31604654  # Scattering factor for B field 1e-6

CR4XA_SIG = 'void(i8, f8[:], f8[:], f8, f8[:], f8[:], f8[:], c16[:], c16[:], f8, c16[:])'
CR4XA_LOCALS = {
    "E0": numba.float64,
    "L": numba.int32,
    "LP": numba.int32,
    "STEP": numba.int8,
    "Z": numba.float64,
}
CR4XA_LOCALS.update((s, numba.complex128) for s in [
    "S1L", "S1LP", "S3", "S3LP", 
    "FS1S1", "FS1S3", "FS3S1", "FS3S3",
    "DELTA", "BL", "GL", "BLP", "GLP", "SSWAP", "DETW",
    "DBB", "DBG", "DGB", "DGG",
    "ES1L", "ENS1L", "ES1LP", "ENS1LP", "ES3L", "ENS3L", "ES3LP", "ENS3LP"])
CR4XA_LOCALS.update(("A{i}{j}".format(i=i, j=j), numba.complex128) for i in range(1,5) for j in range(1,5))
CR4XA_LOCALS.update(("B{i}{j}".format(i=i, j=j), numba.complex128) for i in range(1,5) for j in range(1,5))
CR4XA_LOCALS.update(("C{i}".format(i=i), numba.complex128) for i in range(1,5))

@njit(CR4XA_SIG, parallel=False, cache=True, locals=CR4XA_LOCALS)
def Cr4xa(N, D, SIGMA, IP, RHO, IRHO, RHOM, U1, U3, KZ, Y):
    EPS = 1e-10
    PI4 = np.pi * 4.0e-6

    if (KZ <= -1.e-10):
        L = N-1
        STEP = -1
        SIGMA_OFFSET = -1
    elif (KZ >= 1.e-10):
        L = 0
        STEP = 1
        SIGMA_OFFSET = 0
    else:
        Y[0] = -1.
        Y[1] = 0.
        Y[2] = 0.
        Y[3] = -1.
        return

    #    Changing the target KZ is equivalent to subtracting the fronting
    #    medium SLD.

    # IP = 1 specifies polarization of the incident beam I+
    # IP = -1 specifies polarization of the incident beam I-
    E0 = KZ*KZ + PI4*(RHO[L]+IP*RHOM[L])

    Z = 0.0
    if (N > 1):
        # chi in layer 1
        LP = L + STEP
        # Branch selection:  the -sqrt below for S1 and S3 will be
        #     +Imag for KZ > Kcrit,
        #     -Real for KZ < Kcrit
        # which covers the S1, S3 waves allowed by the boundary conditions in the
        # fronting and backing medium:
        # either traveling forward (+Imag) or decaying exponentially forward (-Real).
        # The decaying exponential only occurs for the transmitted forward wave in the backing:
        # the root +iKz is automatically chosen for the incident wave in the fronting.
        #
        # In the fronting, the -S1 and -S3 waves are either traveling waves backward (-Imag)
        # or decaying along the -z reflection direction (-Real) * (-z) = (+Real*z).
        # NB: This decaying reflection only occurs when the reflected wave is below Kcrit
        # while the incident wave is above Kcrit, so it only happens for spin-flip from
        # minus to plus (lower to higher potential energy) and the observed R-+ will
        # actually be zero at large distances from the interface.
        #
        # In the backing, the -S1 and -S3 waves are explicitly set to be zero amplitude
        # by the boundary conditions (neutrons only incident in the fronting medium - no
        # source of neutrons below).
        #
        S1L = -sqrt(complex(PI4*(RHO[L]+RHOM[L]) - E0, -PI4*(fabs(IRHO[L])+EPS)))
        S3L = -sqrt(complex(PI4*(RHO[L]-RHOM[L]) -
                    E0, -PI4*(fabs(IRHO[L])+EPS)))
        S1LP = -sqrt(complex(PI4*(RHO[LP]+RHOM[LP]) -
                     E0, -PI4*(fabs(IRHO[LP])+EPS)))
        S3LP = -sqrt(complex(PI4*(RHO[LP]-RHOM[LP]) -
                     E0, -PI4*(fabs(IRHO[LP])+EPS)))
        SIGMAL = SIGMA[L+SIGMA_OFFSET]

        if (abs(U1[L]) <= 1.0):
            # then Bz >= 0
            # BL and GL are zero in the fronting.
            pass
        else:
            # then Bz < 0: flip!
            # This is probably impossible, since Bz defines the +z direction
            # in the fronting medium, but just in case...
            SSWAP = S1L
            S1L = S3L
            S3L = SSWAP;  # swap S3 and S1

        if (abs(U1[LP]) <= 1.0):
            # then Bz >= 0
            BLP = U1[LP]
            GLP = 1.0/U3[LP]
        else:
            # then Bz < 0: flip!
            BLP = U3[LP]
            GLP = 1.0/U1[LP]
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP;  # swap S3 and S1

        DELTA = 0.5 / (1.0 - (BLP*GLP))

        FS1S1 = S1L/S1LP
        FS1S3 = S1L/S3LP
        FS3S1 = S3L/S1LP
        FS3S3 = S3L/S3LP

        B11 = DELTA * 1.0 * (1.0 + FS1S1)
        B12 = DELTA * 1.0 * (1.0 - FS1S1) * exp(2.*S1L*S1LP*SIGMAL*SIGMAL)
        B13 = DELTA * -GLP * (1.0 + FS3S1)
        B14 = DELTA * -GLP * (1.0 - FS3S1) * exp(2.*S3L*S1LP*SIGMAL*SIGMAL)

        B21 = DELTA * 1.0 * (1.0 - FS1S1) * exp(2.*S1L*S1LP*SIGMAL*SIGMAL)
        B22 = DELTA * 1.0 * (1.0 + FS1S1)
        B23 = DELTA * -GLP * (1.0 - FS3S1) * exp(2.*S3L*S1LP*SIGMAL*SIGMAL)
        B24 = DELTA * -GLP * (1.0 + FS3S1)

        B31 = DELTA * -BLP * (1.0 + FS1S3)
        B32 = DELTA * -BLP * (1.0 - FS1S3) * exp(2.*S1L*S3LP*SIGMAL*SIGMAL)
        B33 = DELTA * 1.0 * (1.0 + FS3S3)
        B34 = DELTA * 1.0 * (1.0 - FS3S3) * exp(2.*S3L*S3LP*SIGMAL*SIGMAL)

        B41 = DELTA * -BLP * (1.0 - FS1S3) * exp(2.*S1L*S3LP*SIGMAL*SIGMAL)
        B42 = DELTA * -BLP * (1.0 + FS1S3)
        B43 = DELTA * 1.0 * (1.0 - FS3S3) * exp(2.*S3L*S3LP*SIGMAL*SIGMAL)
        B44 = DELTA * 1.0 * (1.0 + FS3S3)

        Z += D[LP]
        L = LP

    #    Process the loop once for each interior layer, either from
    #    front to back or back to front.
    for I in range(1,N-1):
        LP = L + STEP
        S1L = S1LP # copy from the layer before
        S3L = S3LP #
        GL = GLP
        BL = BLP
        S1LP = -sqrt(complex(PI4*(RHO[LP]+RHOM[LP])-E0, -PI4*(fabs(IRHO[LP])+EPS)))
        S3LP = -sqrt(complex(PI4*(RHO[LP]-RHOM[LP])-E0, -PI4*(fabs(IRHO[LP])+EPS)))
        SIGMAL = SIGMA[L+SIGMA_OFFSET]

        if (abs(U1[LP]) <= 1.0):
            # then Bz >= 0
            BLP = U1[LP]
            GLP = 1.0/U3[LP]
        else:
            # then Bz < 0: flip!
            BLP = U3[LP]
            GLP = 1.0/U1[LP]
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP; # swap S3 and S1

        DELTA = 0.5 / (1.0 - (BLP*GLP))
        DBB = (BL - BLP) * DELTA; # multiply by delta here?
        DBG = (1.0 - BL*GLP) * DELTA
        DGB = (1.0 - GL*BLP) * DELTA
        DGG = (GL - GLP) * DELTA

        ES1L = exp(S1L*Z)
        ENS1L = 1.0 / ES1L
        ES1LP = exp(S1LP*Z)
        ENS1LP = 1.0 / ES1LP
        ES3L = exp(S3L*Z)
        ENS3L = 1.0 / ES3L
        ES3LP = exp(S3LP*Z)
        ENS3LP = 1.0 / ES3LP

        FS1S1 = S1L/S1LP
        FS1S3 = S1L/S3LP
        FS3S1 = S3L/S1LP
        FS3S3 = S3L/S3LP

        A11 = A22 = DBG * (1.0 + FS1S1)
        A11 *= ES1L * ENS1LP
        A22 *= ENS1L * ES1LP
        A12 = A21 = DBG * (1.0 - FS1S1) * exp(2.*S1L*S1LP*SIGMAL*SIGMAL)
        A12 *= ENS1L * ENS1LP
        A21 *= ES1L  * ES1LP
        A13 = A24 = DGG * (1.0 + FS3S1)
        A13 *= ES3L  * ENS1LP
        A24 *= ENS3L * ES1LP
        A14 = A23 = DGG * (1.0 - FS3S1) * exp(2.*S3L*S1LP*SIGMAL*SIGMAL)
        A14 *= ENS3L * ENS1LP
        A23 *= ES3L  * ES1LP

        A31 = A42 = DBB * (1.0 + FS1S3)
        A31 *= ES1L * ENS3LP
        A42 *= ENS1L * ES3LP
        A32 = A41 = DBB * (1.0 - FS1S3) * exp(2.*S1L*S3LP*SIGMAL*SIGMAL)
        A32 *= ENS1L * ENS3LP
        A41 *= ES1L  * ES3LP
        A33 = A44 = DGB * (1.0 + FS3S3)
        A33 *= ES3L * ENS3LP
        A44 *= ENS3L * ES3LP
        A34 = A43 = DGB * (1.0 - FS3S3) * exp(2.*S3L*S3LP*SIGMAL*SIGMAL)
        A34 *= ENS3L * ENS3LP
        A43 *= ES3L * ES3LP


        #    Matrix update B=A*B
        C1=A11*B11+A12*B21+A13*B31+A14*B41
        C2=A21*B11+A22*B21+A23*B31+A24*B41
        C3=A31*B11+A32*B21+A33*B31+A34*B41
        C4=A41*B11+A42*B21+A43*B31+A44*B41
        B11=C1
        B21=C2
        B31=C3
        B41=C4

        C1=A11*B12+A12*B22+A13*B32+A14*B42
        C2=A21*B12+A22*B22+A23*B32+A24*B42
        C3=A31*B12+A32*B22+A33*B32+A34*B42
        C4=A41*B12+A42*B22+A43*B32+A44*B42
        B12=C1
        B22=C2
        B32=C3
        B42=C4

        C1=A11*B13+A12*B23+A13*B33+A14*B43
        C2=A21*B13+A22*B23+A23*B33+A24*B43
        C3=A31*B13+A32*B23+A33*B33+A34*B43
        C4=A41*B13+A42*B23+A43*B33+A44*B43
        B13=C1
        B23=C2
        B33=C3
        B43=C4

        C1=A11*B14+A12*B24+A13*B34+A14*B44
        C2=A21*B14+A22*B24+A23*B34+A24*B44
        C3=A31*B14+A32*B24+A33*B34+A34*B44
        C4=A41*B14+A42*B24+A43*B34+A44*B44
        B14=C1
        B24=C2
        B34=C3
        B44=C4

        Z += D[LP]
        L = LP

    #    Done computing B = A(N)*...*A(2)*A(1)*I
    DETW = B44*B22 - B24*B42


    #    Calculate reflectivity coefficients specified by POLSTAT
    # IP = +1 fills in ++, +-, -+, --; IP = -1 only fills in -+, --.
    if IP > 0:
        Y[0] = (B24*B41 - B21*B44)/DETW # ++
        Y[1] = (B21*B42 - B41*B22)/DETW # +-
    Y[2] = (B24*B43 - B23*B44)/DETW # -+
    Y[3] = (B23*B42 - B43*B22)/DETW # --

#Cr4xa.inspect_types()

#@cc.export('mag_amplitude')
MAGAMP_SIG = 'void(f8[:], f8[:], f8[:], f8[:], f8[:], c16[:], c16[:], f8[:], c16[:,:])'
@njit(MAGAMP_SIG, parallel=False, cache=True)
def magnetic_amplitude_py(d, sigma, rho, irho, rhoM, u1, u3, KZ, R):
    """
    python version of calculation
    implicit returns: Ra, Rb, Rc, Rd
    """
    #assert rho_index is None
    layers = len(d)
    points = len(KZ)
    if (np.fabs(rhoM[0]) <= MINIMAL_RHO_M and np.fabs(rhoM[layers-1]) <= MINIMAL_RHO_M):
        # calculations for I+ and I- are the same in the fronting and backing.
        for i in prange(points):
            Cr4xa(layers, d, sigma, 1.0, rho, irho, rhoM, u1, u3, KZ[i], R[i])
    else:
        # plus polarization must be before minus polarization because it
        # fills in all R++, R+-, R-+, R--, but minus polarization only fills
        # in R-+, R--.
        for i in prange(points):
            Cr4xa(layers, d, sigma, 1.0, rho, irho, rhoM, u1, u3, KZ[i], R[i])

        # minus polarization
        for i in prange(points):
            Cr4xa(layers, d, sigma, -1.0, rho, irho, rhoM, u1, u3, KZ[i], R[i])

#try:
#    from .magnetic_amplitude import mag_amplitude as magnetic_amplitude_py
#    print("loaded from compiled module")
#except ImportError:
#    from numba.pycc import CC
#    cc = CC('magnetic_amplitude')
#    Cr4xa = cc.export('Cr4xa')(Cr4xa)
#    magnetic_amplitude_py = cc.export('magnetic_amplitude')(magnetic_amplitude_py)
#    print('could not load from compiled module, building for next time...')
#    cc.compile()
#    magnetic_amplitude_py = _magnetic_amplitude_py


@njit('(f8[:], f8[:], f8[:], f8[:], f8[:])', cache=True, parallel=False)
def _convolve_uniform(xi, yi, x, dx, y):
    root_12_over_2 = np.sqrt(3)
    left_index = 0
    N_xi = len(xi)
    N_x = len(x)
    for k in prange(N_x):
        x_k = x[k]
        # Convert 1-sigma width to 1/2 width of the region
        limit = dx[k] * root_12_over_2
        # print(f"point {x_k} +/- {limit}")
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
        while left_index < N_xi-2 and xi[left_index] < left:
            left_index += 1
        while left_index > 0 and xi[left_index] > left:
            left_index -= 1

        # Set the first interval.
        total = 0.
        right_index = left_index + 1
        x1, y1 = xi[left_index], yi[left_index]
        x2, y2 = xi[right_index], yi[right_index]


        # Subtract the excess from left interval before the left edge.
        # print(f" left {left} in {(x1, y1)}, {(x2, y2)}")
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
            # print(f" left correction {area}")

        # Do trapezoidal integration up to and including the end interval
        while right_index < N_xi-1 and x2 < right:
            # Add the current interval if it isn't empty
            if x1 != x2:
                area = 0.5*(y1 + y2)*(x2 - x1)
                total += area
                # print(f" adding {(x1,y1)}, {(x2, y2)} as {area}")
            # Move to the next interval
            right_index += 1
            x1, y1, x2, y2 = x2, y2, xi[right_index], yi[right_index]
        if x1 != x2:
            area = 0.5*(y1 + y2)*(x2 - x1)
            total += area
            # print(f" adding final {(x1,y1)}, {(x2, y2)} as {area}")

        # Subtract the excess from the right interval after the right edge.
        # print(f" right {right} in {(x1, y1)}, {(x2, y2)}")
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
            # print(f" right correction {area}")

        # Normalize by interval length
        if left < right:
            # print(f" normalize by length {right} - {left}")
            y[k] = total / (right - left)
        elif x1 < x2:
            # If dx = 0 using the value interpolated at x (with left=right=x).
            # print(f" dirac delta at {left} = {right} in {(x1, y1)}, {(x2, y2)}")
            offset = left - x1
            slope = (y2 - y1)/(x2 - x1)
            y[k] = y1 + slope*offset
        else:
            # At an empty interval in the theory function. Average the y.
            # print(f" empty interval with {left} = {right} in {(x1, y1)}, {(x2, y2)}")
            y[k] = 0.5*(y1 + y2)

PI4 =          12.56637061435917295385
PI_180 =        0.01745329251994329576
LN256 =         5.54517744447956247533
SQRT2 =         1.41421356237309504880
SQRT2PI =       2.50662827463100050241
LOG_RESLIMIT = -6.90775527898213703123

from math import erf
@njit('f8(f8[:], f8[:], i8, i8, f8, f8, f8)', cache=True, parallel=False, locals={
    "z": numba.float64,
    "Glo": numba.float64,
    "erflo": numba.float64,
    "erfmin": numba.float64,
    "y": numba.float64,
    "zhi": numba.float64,
    "Ghi": numba.float64,
    "erfhi": numba.float64,
    "m": numba.float64,
    "b": numba.float64,
})
def convolve_gaussian_point(xin, yin, k, n,
               xo, limit, sigma):

    two_sigma_sq = 2. * sigma * sigma
    #double z, Glo, erflo, erfmin, y

    z = xo - xin[k]
    Glo = exp(-z*z/two_sigma_sq)
    erfmin = erflo = erf(-z/(SQRT2*sigma))
    y = 0.
    #/* printf("%5.3f: (%5.3f,%11.5g)",xo,xin[k],yin[k]); */
    while (k < n):
        k += 1
        if (xin[k] != xin[k-1]):
            #/* No additional contribution from duplicate points. */

            #/* Compute the next endpoint */
            zhi = xo - xin[k]
            Ghi = exp(-zhi*zhi/two_sigma_sq)
            erfhi = erf(-zhi/(SQRT2*sigma))
            m = (yin[k]-yin[k-1])/(xin[k]-xin[k-1])
            b = yin[k] - m * xin[k]

            #/* Add the integrals. */
            y += 0.5*(m*xo+b)*(erfhi-erflo) - sigma/SQRT2PI*m*(Ghi-Glo)

            #/* Debug computation failures. */
            # if isnan(y) {
            #     print("NaN from %d: zhi=%g, Ghi=%g, erfhi=%g, m=%g, b=%g\n",
            #          % (k,zhi,Ghi,erfhi,m,b))
            # }

            #/* Save the endpoint for next trapezoid. */
            Glo = Ghi
            erflo = erfhi

            #/* Check if we've calculated far enough */
            if (xin[k] >= xo+limit):
                break

    #/* printf(" (%5.3f,%11.5g)",xin[k<n?k:n-1],yin[k<n?k:n-1]); */

    #/* Normalize by the area of the truncated gaussian */
    #/* At this point erflo = erfmax */
    #/* printf ("---> %11.5g\n",2*y/(erflo-erfmin)); */
    return 2 * y / (erflo - erfmin)


# has same performance when using guvectorize instead of njit:
#@numba.guvectorize("(i8, f8[:], f8[:], i8, f8[:], f8[:], f8[:])", '(),(m),(m),(),(n),(n)->(n)')

@njit("(i8, f8[:], f8[:], i8, f8[:], f8[:], f8[:])", cache=True, parallel=False, locals={
    "sigma": numba.float64,
    "xo": numba.float64,
    "limit": numba.float64,
    "k_in": numba.int64,
    "k_out": numba.int64,
})
def _convolve_gaussian( Nin, xin, yin, Nout, x, dx, y):
    # size_t in,out;

    #/* FIXME fails if xin are not sorted; slow if x not sorted */
    assert(Nin>1)

    #/* Scan through all x values to be calculated */
    #/* Re: omp, each thread is going through the entire input array,
    # * independently, computing the resolution from the neighbourhood
    # * around its individual output points.  The firstprivate(in)
    # * clause sets each thread to keep its own copy of in, initialized
    # * at in's initial value of zero.  The "schedule(static,1)" clause
    # * puts neighbouring points in separate threads, which is a benefit
    # * since there will be less backtracking if resolution width increases
    # * from point to point.  Because the schedule is static, this does not
    # * significantly increase the parallelization overhead.  Because the
    # * threads are operating on interleaved points, there should be fewer cache
    # * misses than if each thread were given different stretches of x to
    # * convolve.
    # */
    k_in = 0
  
    for k_out in range(Nout):
        #/* width of resolution window for x is w = 2 dx^2. */
        sigma = dx[k_out]
        xo = x[k_out]
        limit = sqrt(-2.*sigma*sigma* LOG_RESLIMIT)

        #// if (out%20==0)

        # /* Line up the left edge of the convolution window */
        # /* It is probably forward from the current position, */
        # /* but if the next dx is a lot higher than the current */
        # /* dx or if the x are not sorted, then it may be before */
        # /* the current position. */
        # /* FIXME verify that the convolution window is just right */
        while (k_in < Nin-1 and xin[k_in] < xo-limit):
            k_in += 1
        while (k_in > 0 and xin[k_in] > xo-limit):
            k_in -=1

        #/* Special handling to avoid 0/0 for w=0. */
        if (sigma > 0.):
            y[k_out] = convolve_gaussian_point(xin,yin,k_in,Nin,xo,limit,sigma)
        elif (k_in < Nin-1):
            #/* Linear interpolation */
            m = (yin[k_in+1]-yin[k_in])/(xin[k_in+1]-xin[k_in])
            b = yin[k_in] - m*xin[k_in]
            y[k_out] = m*xo + b
        elif (k_in > 0):
            #/* Linear extrapolation */
            m = (yin[k_in]-yin[k_in-1])/(xin[k_in]-xin[k_in-1])
            b = yin[k_in] - m*xin[k_in]
            y[k_out] = m*xo + b
        else:
            #/* Can't happen because there is more than one point in xin. */
            assert(Nin>1)

def _convolve_gaussian_vector( Nin, xin, yin, xo, sigma):
    # size_t in,out;

    #/* FIXME fails if xin are not sorted; slow if x not sorted */
    assert(Nin>1)

    #/* Scan through all x values to be calculated */
    #/* Re: omp, each thread is going through the entire input array,
    # * independently, computing the resolution from the neighbourhood
    # * around its individual output points.  The firstprivate(in)
    # * clause sets each thread to keep its own copy of in, initialized
    # * at in's initial value of zero.  The "schedule(static,1)" clause
    # * puts neighbouring points in separate threads, which is a benefit
    # * since there will be less backtracking if resolution width increases
    # * from point to point.  Because the schedule is static, this does not
    # * significantly increase the parallelization overhead.  Because the
    # * threads are operating on interleaved points, there should be fewer cache
    # * misses than if each thread were given different stretches of x to
    # * convolve.
    # */
    k_in = 0
  
    for k_out in prange(Nout):
        #/* width of resolution window for x is w = 2 dx^2. */
        sigma = dx[k_out]
        xo = x[k_out]
        limit = sqrt(-2.*sigma*sigma* LOG_RESLIMIT)

        #// if (out%20==0)

        # /* Line up the left edge of the convolution window */
        # /* It is probably forward from the current position, */
        # /* but if the next dx is a lot higher than the current */
        # /* dx or if the x are not sorted, then it may be before */
        # /* the current position. */
        # /* FIXME verify that the convolution window is just right */
        while (k_in < Nin-1 and xin[k_in] < xo-limit):
            k_in += 1
        while (k_in > 0 and xin[k_in] > xo-limit):
            k_in -=1

        #/* Special handling to avoid 0/0 for w=0. */
        if (sigma > 0.):
            y[k_out] = convolve_gaussian_point(xin,yin,k_in,Nin,xo,limit,sigma)
        elif (k_in < Nin-1):
            #/* Linear interpolation */
            m = (yin[k_in+1]-yin[k_in])/(xin[k_in+1]-xin[k_in])
            b = yin[k_in] - m*xin[k_in]
            y[k_out] = m*xo + b
        elif (k_in > 0):
            #/* Linear extrapolation */
            m = (yin[k_in]-yin[k_in-1])/(xin[k_in]-xin[k_in-1])
            b = yin[k_in] - m*xin[k_in]
            y[k_out] = m*xo + b
        else:
            #/* Can't happen because there is more than one point in xin. */
            assert(Nin>1)

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
        #reflmodule.convolve(xi, yi, x, dx, y)
        _convolve_gaussian(len(xi), xi, yi, len(x), x, dx, y)
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
    xi=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yi=[1, 3, 1, 2, 7, 3, 1, 2, 1, 3]
    _check_uniform("uniform aligned", xi, yi, x = [2, 4, 6, 8], dx = 1)
    _check_uniform("uniform unaligned", xi, yi,
                   x = [2.5, 4.5, 6.5, 8.5], dx = 1)
    _check_uniform("uniform wide", xi, yi, x = [2.5, 4.5, 6.5, 8.5], dx = 3)
    # Check bad values
    ystar = convolve(xi, yi, x = [-3, 13], dx = [1/np.sqrt(3)]*2, resolution ='uniform')
    assert (ystar == 0.).all()

    xi=[1.1, 2.3, 2.8, 4.2, 5, 6, 7, 8, 9, 10]
    yi=[1, 3, 1, 2, 7, 3, 1, 2, 1, 3]
    _check_uniform("uniform unaligned", xi, yi,
                   x = [2.5, 4.5, 6.5, 8.5], dx = 1)

def _check_uniform(name, xi, yi, x, dx):
    # Note: using fixed dx since that's all _check_spline supports.
    ystar=convolve(xi, yi, x, [dx/np.sqrt(3)]*len(x), resolution = 'uniform')
    # print("xi", xi)
    # print("yi", yi)
    # print("x", x, "dx", dx)
    # print("ystar", ystar)
    xp=[-dx, -dx, dx, dx]
    yp=[0, 0.5/dx, 0.5/dx, 0]
    _check_spline(name, xi, yi, xp, yp, x, ystar)

def test_convolve_sampled():
    xi=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yi=[1, 3, 1, 2, 1, 3, 1, 2, 1, 3]
    xp=[-1, 0, 1, 2, 3]
    yp=[1, 4, 3, 2, 1]
    _check_sampled("sampled aligned", xi, yi, xp, yp, dx = 1)
    _check_sampled("sampled unaligned", xi, yi,
                   _dense(xp) - 0.2000003, yp, dx = 1)
    _check_sampled("sampled wide", xi, yi, xp, yp, dx = 2)
    _check_sampled("sampled super wide", xi, yi, xp, yp, dx = 10)

def _check_sampled(name, xi, yi, xp, yp, dx):
    ystar= convolve_sampled(xi, yi, xp, yp, xi, dx = np.full_like(xi, dx))
    xp=np.array(xp)*dx
    _check_spline(name, xi, yi, xp, yp, xi, ystar)

def _check_spline(name, xi, yi, xp, yp, x, ystar):
    step=0.0001
    xpfine=np.arange(xp[0], xp[-1] + step / 10, step)
    ypfine=np.interp(xpfine, xp, yp)
    # make sure xfine is wide enough by adding an extra interval at the ends
    xfine=np.arange(xi[0] + xpfine[0] - 2*step,
                    xi[-1] + xpfine[-1] + 2*step, step)
    yfine = np.interp(xfine, xi, yi, left = 0, right =0)
    pidx=np.searchsorted(xfine, np.array(x) + xp[0])
    left, right=np.searchsorted(xfine, [xi[0], xi[-1]])

    conv=[]
    for pk in pidx:
        norm_start=max(0, left - pk)
        norm_end=min(len(xpfine), right - pk)
        norm=step * np.sum(ypfine[norm_start:norm_end])
        conv.append(step * np.sum(ypfine * yfine[pk:pk + len(xpfine)]) / norm)

    # print("checking convolution %s"%(name, ))
    # print(" ".join("%7.4f"%yi for yi in ystar))
    # print(" ".join("%7.4f"%yi for yi in conv))
    assert all(abs(yi - fi) < 0.0005 for (yi, fi) in zip(ystar, conv)), name
