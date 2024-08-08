import numpy as np
import cmath
import numba
from numba import cuda

from .clone_module import clone_module

MODULE = clone_module('refl1d.lib.python.magnetic')

MODULE.prange = numba.prange

calculate_U1_U3_single = numba.njit(cache=True)(MODULE.calculate_U1_U3_single)
MODULE.calculate_U1_U3_single = calculate_U1_U3_single

calculate_u1_u3 = numba.njit(cache=True)(MODULE.calculate_u1_u3)
MODULE.calculate_u1_u3 = calculate_u1_u3

import sys
from math import fabs
# from numpy import pi, sin, cos, sqrt, exp, conj, radians

EPS = sys.float_info.epsilon
M_PI = np.pi
PI4 = 4.0e-6 * np.pi
B2SLD = 2.31604654  # Scattering factor for B field 1e-6
MINIMAL_RHO_M = 1e-2  # in units of 1e-6/A^2

DEBUG = False

CR4XA_SIG = 'void(i8, f8[:], f8[:], f8, f8[:], f8[:], f8[:], c16[:], c16[:], f8, i4, c16[:], c16[:], c16[:], c16[:])'
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
CR4XA_LOCALS.update(("A{i}{j}".format(i=i, j=j), numba.complex128)
                    for i in range(1, 5) for j in range(1, 5))
CR4XA_LOCALS.update(("B{i}{j}".format(i=i, j=j), numba.complex128)
                    for i in range(1, 5) for j in range(1, 5))
CR4XA_LOCALS.update(("C{i}".format(i=i), numba.complex128)
                    for i in range(1, 5))

@cuda.jit(CR4XA_SIG, device=True, cache=True)
def Cr4xa(N, D, SIGMA, IP, RHO, IRHO, RHOM, U1, U3, KZ, POINT, YA, YB, YC, YD):
    EPS = 1e-10

    if (KZ <= -1.e-10):
        L = N-1
        STEP = -1
        SIGMA_OFFSET = -1
    elif (KZ >= 1.e-10):
        L = 0
        STEP = 1
        SIGMA_OFFSET = 0
    else:
        YA[POINT] = -1.
        YB[POINT] = 0.
        YC[POINT] = 0.
        YD[POINT] = -1.
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

        RHO_L = RHO[L]
        RHOM_L = RHOM[L]
        IRHO_L = IRHO[L]
        U1_L = U1[L]
        U3_L = U3[L]

        if POINT == 0:
            print(D[L], SIGMA[L], RHO_L, IRHO_L, RHOM_L)


        RHO_LP = RHO[LP]
        RHOM_LP = RHOM[LP]
        IRHO_LP = IRHO[LP]
        U1_LP = U1[LP]
        U3_LP = U3[LP]

        S1L = -cmath.sqrt(complex(PI4*(RHO_L+RHOM_L) -
                    E0, -PI4*(fabs(IRHO_L)+EPS)))
        S3L = -cmath.sqrt(complex(PI4*(RHO_L-RHOM_L) -
                    E0, -PI4*(fabs(IRHO_L)+EPS)))
        S1LP = -cmath.sqrt(complex(PI4*(RHO_LP+RHOM_LP) -
                     E0, -PI4*(fabs(IRHO_LP)+EPS)))
        S3LP = -cmath.sqrt(complex(PI4*(RHO_LP-RHOM_LP) -
                     E0, -PI4*(fabs(IRHO_LP)+EPS)))
        SIGMAL = SIGMA[L+SIGMA_OFFSET]

        if (abs(U1_L) <= 1.0):
            # then Bz >= 0
            # BL and GL are zero in the fronting.
            pass
        else:
            # then Bz < 0: flip!
            # This is probably impossible, since Bz defines the +z direction
            # in the fronting medium, but just in case...
            SSWAP = S1L
            S1L = S3L
            S3L = SSWAP  # swap S3 and S1

        if (abs(U1_LP) <= 1.0):
            # then Bz >= 0
            BLP = U1_LP
            GLP = 1.0/U3_LP
        else:
            # then Bz < 0: flip!
            BLP = U3_LP
            GLP = 1.0/U1_LP
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP  # swap S3 and S1

        DELTA = 0.5 / (1.0 - (BLP*GLP))

        FS1S1 = S1L/S1LP
        FS1S3 = S1L/S3LP
        FS3S1 = S3L/S1LP
        FS3S3 = S3L/S3LP

        B11 = DELTA * 1.0 * (1.0 + FS1S1)
        B12 = DELTA * 1.0 * (1.0 - FS1S1) * cmath.exp(2.*S1L*S1LP*SIGMAL*SIGMAL)
        B13 = DELTA * -GLP * (1.0 + FS3S1)
        B14 = DELTA * -GLP * (1.0 - FS3S1) * cmath.exp(2.*S3L*S1LP*SIGMAL*SIGMAL)

        B21 = DELTA * 1.0 * (1.0 - FS1S1) * cmath.exp(2.*S1L*S1LP*SIGMAL*SIGMAL)
        B22 = DELTA * 1.0 * (1.0 + FS1S1)
        B23 = DELTA * -GLP * (1.0 - FS3S1) * cmath.exp(2.*S3L*S1LP*SIGMAL*SIGMAL)
        B24 = DELTA * -GLP * (1.0 + FS3S1)

        B31 = DELTA * -BLP * (1.0 + FS1S3)
        B32 = DELTA * -BLP * (1.0 - FS1S3) * cmath.exp(2.*S1L*S3LP*SIGMAL*SIGMAL)
        B33 = DELTA * 1.0 * (1.0 + FS3S3)
        B34 = DELTA * 1.0 * (1.0 - FS3S3) * cmath.exp(2.*S3L*S3LP*SIGMAL*SIGMAL)

        B41 = DELTA * -BLP * (1.0 - FS1S3) * cmath.exp(2.*S1L*S3LP*SIGMAL*SIGMAL)
        B42 = DELTA * -BLP * (1.0 + FS1S3)
        B43 = DELTA * 1.0 * (1.0 - FS3S3) * cmath.exp(2.*S3L*S3LP*SIGMAL*SIGMAL)
        B44 = DELTA * 1.0 * (1.0 + FS3S3)

        Z += D[LP]
        L = LP

    cuda.syncthreads()
    #    Process the loop once for each interior layer, either from
    #    front to back or back to front.
    for I in range(1, N-1):
        LP = L + STEP
        S1L = S1LP  # copy from the layer before
        S3L = S3LP
        GL = GLP
        BL = BLP

        RHO_LP = RHO[LP]
        # cuda.syncthreads()
        RHOM_LP = RHOM[LP]
        # cuda.syncthreads()
        IRHO_LP = IRHO[LP]
        # cuda.syncthreads()
        U1_LP = U1[LP]
        # cuda.syncthreads()
        U3_LP = U3[LP]
        # cuda.syncthreads()

        S1LP = -cmath.sqrt(complex(PI4*(RHO_LP+RHOM_LP)-E0,
                             -PI4*(fabs(IRHO_LP)+EPS)))
        S3LP = -cmath.sqrt(complex(PI4*(RHO_LP-RHOM_LP)-E0,
                             -PI4*(fabs(IRHO_LP)+EPS)))
        SIGMAL = SIGMA[L+SIGMA_OFFSET]

        if (abs(U1_LP) <= 1.0):
            # then Bz >= 0
            BLP = U1_LP
            GLP = 1.0/U3_LP
        else:
            # then Bz < 0: flip!
            BLP = U3_LP
            GLP = 1.0/U1_LP
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP  # swap S3 and S1
        cuda.syncthreads()


        DELTA = 0.5 / (1.0 - (BLP*GLP))
        DBB = (BL - BLP) * DELTA  # multiply by delta here?
        DBG = (1.0 - BL*GLP) * DELTA
        DGB = (1.0 - GL*BLP) * DELTA
        DGG = (GL - GLP) * DELTA

        ES1L = cmath.exp(S1L*Z)
        ENS1L = 1.0 / ES1L
        ES1LP = cmath.exp(S1LP*Z)
        ENS1LP = 1.0 / ES1LP
        ES3L = cmath.exp(S3L*Z)
        ENS3L = 1.0 / ES3L
        ES3LP = cmath.exp(S3LP*Z)
        ENS3LP = 1.0 / ES3LP

        FS1S1 = S1L/S1LP
        FS1S3 = S1L/S3LP
        FS3S1 = S3L/S1LP
        FS3S3 = S3L/S3LP

        A11 = A22 = DBG * (1.0 + FS1S1)
        A11 *= ES1L * ENS1LP
        A22 *= ENS1L * ES1LP
        A12 = A21 = DBG * (1.0 - FS1S1) * cmath.exp(2.*S1L*S1LP*SIGMAL*SIGMAL)
        A12 *= ENS1L * ENS1LP
        A21 *= ES1L * ES1LP
        A13 = A24 = DGG * (1.0 + FS3S1)
        A13 *= ES3L * ENS1LP
        A24 *= ENS3L * ES1LP
        A14 = A23 = DGG * (1.0 - FS3S1) * cmath.exp(2.*S3L*S1LP*SIGMAL*SIGMAL)
        A14 *= ENS3L * ENS1LP
        A23 *= ES3L * ES1LP

        A31 = A42 = DBB * (1.0 + FS1S3)
        A31 *= ES1L * ENS3LP
        A42 *= ENS1L * ES3LP
        A32 = A41 = DBB * (1.0 - FS1S3) * cmath.exp(2.*S1L*S3LP*SIGMAL*SIGMAL)
        A32 *= ENS1L * ENS3LP
        A41 *= ES1L * ES3LP
        A33 = A44 = DGB * (1.0 + FS3S3)
        A33 *= ES3L * ENS3LP
        A44 *= ENS3L * ES3LP
        A34 = A43 = DGB * (1.0 - FS3S3) * cmath.exp(2.*S3L*S3LP*SIGMAL*SIGMAL)
        A34 *= ENS3L * ENS3LP
        A43 *= ES3L * ES3LP

        #    Matrix update B=A*B
        C1 = A11*B11+A12*B21+A13*B31+A14*B41
        C2 = A21*B11+A22*B21+A23*B31+A24*B41
        C3 = A31*B11+A32*B21+A33*B31+A34*B41
        C4 = A41*B11+A42*B21+A43*B31+A44*B41
        B11 = C1
        B21 = C2
        B31 = C3
        B41 = C4

        C1 = A11*B12+A12*B22+A13*B32+A14*B42
        C2 = A21*B12+A22*B22+A23*B32+A24*B42
        C3 = A31*B12+A32*B22+A33*B32+A34*B42
        C4 = A41*B12+A42*B22+A43*B32+A44*B42
        B12 = C1
        B22 = C2
        B32 = C3
        B42 = C4

        C1 = A11*B13+A12*B23+A13*B33+A14*B43
        C2 = A21*B13+A22*B23+A23*B33+A24*B43
        C3 = A31*B13+A32*B23+A33*B33+A34*B43
        C4 = A41*B13+A42*B23+A43*B33+A44*B43
        B13 = C1
        B23 = C2
        B33 = C3
        B43 = C4

        C1 = A11*B14+A12*B24+A13*B34+A14*B44
        C2 = A21*B14+A22*B24+A23*B34+A24*B44
        C3 = A31*B14+A32*B24+A33*B34+A34*B44
        C4 = A41*B14+A42*B24+A43*B34+A44*B44
        B14 = C1
        B24 = C2
        B34 = C3
        B44 = C4

        Z += D[LP]
        cuda.syncthreads()
        L = LP

    #    Done computing B = A(N)*...*A(2)*A(1)*I
    DETW = B44*B22 - B24*B42

    #    Calculate reflectivity coefficients specified by POLSTAT
    # IP = +1 fills in ++, +-, -+, --; IP = -1 only fills in -+, --.
    if IP > 0:
        YA[POINT] = (B24*B41 - B21*B44)/DETW  # ++
        YB[POINT] = (B21*B42 - B41*B22)/DETW  # +-
    YC[POINT] = (B24*B43 - B23*B44)/DETW  # -+
    YD[POINT] = (B23*B42 - B43*B22)/DETW  # --
    if POINT == 0:
            print(SIGMAL)

# CR4XA_ALIGNED_SIG = 'void(i8, f8[:], f8, f8, i4, c16[:], c16[:], c16[:], c16[:])'
CR4XA_ALIGNED_SIG = 'void(i4, f4[:], i4, f4, i4, i4, c16[:], c16[:], c16[:], c16[:])'

##################################################################################
# fp32 complex calculations to operate on complex tuples (real: fp32, imag: fp32)
##################################################################################

@cuda.jit(device=True)
def mulcx(a, b):
    # multiply complex numbers a(r,i) and b(r,i)
    ar, ai = a
    br, bi = b
    r = ar * br - ai * bi
    i = ar * bi + ai * br
    return r, i

# import math

@cuda.jit('UniTuple(float32, 2)(UniTuple(float32, 2))', device=True)
def nsqrtcx(a):
    # negative complex sqrt of a = complex(ar,ai)
    ar, ai = a
    # halfth = math.atan2(ai, ar) / 2.0
    halfth = cuda.libdevice.atan2f(ai, ar) / numba.float32(2.0)
    s, c = cuda.libdevice.fast_sincosf(halfth)
    rsq = ar*ar + ai*ai
    # sqrtr = math.pow(rsq, 0.25)
    nsqrtr = -cuda.libdevice.fast_powf(rsq, numba.float32(0.25))
    # return np.float32(-sqrtr * math.cos(halfth)), np.float32(-sqrtr * math.sin(halfth))
    return nsqrtr * c, nsqrtr * s

@cuda.jit('UniTuple(float32, 2)(UniTuple(float32, 2))', device=True)
def expcx(a):
    """ complex exponential exp(complex(ar, ai)) """
    ar, ai = a
    s, c = cuda.libdevice.fast_sincosf(ai)
    r = cuda.libdevice.expf(ar)
    return r * c, r * s

@cuda.jit('float32(UniTuple(float32, 2))', device=True)
def abscx(a):
    """ absolute value of complex(ar, ai) """
    ar, ai = a
    # return np.float32(math.sqrt(ar * ar + ai * ai))
    return cuda.libdevice.sqrtf(ar * ar + ai * ai)

@cuda.jit('UniTuple(float32, 2)(UniTuple(float32, 2))', device=True)
def invcx(a):
    """ 1 / complex(ar, ai) """
    ar, ai = a
    denominator = ar * ar + ai * ai
    return ar / denominator, -ai / denominator

@cuda.jit('UniTuple(float32, 2)(UniTuple(float32, 2), UniTuple(float32, 2))', device=True)
def divcx(a, b):
    """ complex(ar, ai) / complex(br, bi) """
    return mulcx(a, invcx(b))

@cuda.jit('UniTuple(float32, 2)(UniTuple(float32, 2), UniTuple(float32, 2))', device=True)
def diffcx(a, b):
    ar, ai = a
    br, bi = b
    return ar - br, ai - bi

@cuda.jit('UniTuple(float32, 2)(UniTuple(float32, 2), UniTuple(float32, 2))', device=True)
def sumcx(a, b):
    ar, ai = a
    br, bi = b
    return ar + br, ai + bi

@cuda.jit('complex128(UniTuple(float32, 2))', device=True)
def cplx(a):
    ar, ai = a
    return complex(ar, ai)

@cuda.jit(device=True)
def isnan(x):
    return x != x

@cuda.jit(CR4XA_ALIGNED_SIG, device=True, inline=True, fastmath=True)
def Cr4xa_aligned(N, layer_params, IP, KZ, POINT, MAXPOINT, YA, YB, YC, YD):
    EPS = 1e-10

    if (KZ <= -1.e-10):
        L = (N-1) * 9
        STEP = -9
        SIGMA_OFFSET = -1
    elif (KZ >= 1.e-10):
        L = 0
        STEP = 9
        SIGMA_OFFSET = 0
    else:
        YA[POINT] = -1.
        YB[POINT] = 0.
        YC[POINT] = 0.
        YD[POINT] = -1.
        return

    #    Changing the target KZ is equivalent to subtracting the fronting
    #    medium SLD.

    # IP = 1 specifies polarization of the incident beam I+
    # IP = -1 specifies polarization of the incident beam I-
    # layer params: d, padded_sigma, rho, irho, rhoM, u1, u3

    shared_layer = cuda.shared.array(shape=(9,), dtype=numba.float32)

    tx = cuda.threadIdx.x
    if tx < 9:
        shared_layer[tx] = layer_params[L + tx]

    cuda.syncthreads()

    D_LP = shared_layer[0]
    SIGMA_LP = shared_layer[1]
    RHO_LP = shared_layer[2]
    IRHO_LP = shared_layer[3]
    RHOM_LP = shared_layer[4]
    U1_LP = complex(shared_layer[5], shared_layer[6])
    U3_LP = complex(shared_layer[7], shared_layer[8])

    if POINT > 996:
        print(POINT, D_LP, SIGMA_LP, RHO_LP, IRHO_LP, RHOM_LP, U1_LP.real, U1_LP.imag, U3_LP.real, U3_LP.imag)


    # if POINT == 0:
        # print(POINT, N, L, STEP, D_LP, SIGMA_LP, RHO_LP, IRHO_LP, RHOM_LP, U1_LP.real, U1_LP.imag, U3_LP.real, U3_LP.imag)

    E0 = KZ*KZ + PI4*(RHO_LP+IP*RHOM_LP)

    Z = 0.0
    S1LP = -cmath.sqrt(complex(PI4*(RHO_LP+RHOM_LP)-E0, -PI4*(fabs(IRHO_LP)+EPS)))
    S3LP = -cmath.sqrt(complex(PI4*(RHO_LP-RHOM_LP)-E0, -PI4*(fabs(IRHO_LP)+EPS)))
    BLP = GLP = 0.0

    if (abs(U1_LP) <= 1.0):
        # then Bz >= 0
        # BL and GL are zero in the fronting.
        pass
    else:
        # then Bz < 0: flip!
        # This is probably impossible, since Bz defines the +z direction
        # in the fronting medium, but just in case...
        SSWAP = S1LP
        S1LP = S3LP
        S3LP = SSWAP  # swap S3 and S1

    # Initialize B matrix (identity)
    B11 = B22 = B33 = B44 = 1.0
    B12 = B13 = B14 = 0
    B21 = B23 = B24 = 0
    B31 = B32 = B34 = 0
    B41 = B42 = B43 = 0

    #    Process the loop once for each interior layer, either from
    #    front to back or back to front.
    for I in range(0, N-1):
        LP = L + STEP
        S1L = S1LP  # copy from the layer before
        S3L = S3LP
        GL = GLP
        BL = BLP

        if tx < 9:
            shared_layer[tx] = layer_params[LP + tx]

        cuda.syncthreads()
        D_LP = shared_layer[0]
        SIGMA_L = shared_layer[1]
        RHO_LP = shared_layer[2]
        IRHO_LP = shared_layer[3]
        RHOM_LP = shared_layer[4]
        U1_LP = complex(shared_layer[5], shared_layer[6])
        U3_LP = complex(shared_layer[7], shared_layer[8])

        S1LP = -cmath.sqrt(complex(PI4*(RHO_LP+RHOM_LP)-E0,
                             -PI4*(fabs(IRHO_LP)+EPS)))
        S3LP = -cmath.sqrt(complex(PI4*(RHO_LP-RHOM_LP)-E0,
                             -PI4*(fabs(IRHO_LP)+EPS)))
        if (abs(U1_LP) <= 1.0):
            # then Bz >= 0
            BLP = U1_LP
            GLP = 1.0/U3_LP
        else:
            # then Bz < 0: flip!
            BLP = U3_LP
            GLP = 1.0/U1_LP
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP  # swap S3 and S1

        DELTA = 0.5 / (1.0 - (BLP*GLP))
        DBB = (BL - BLP) * DELTA  # multiply by delta here?
        DBG = (1.0 - BL*GLP) * DELTA
        DGB = (1.0 - GL*BLP) * DELTA
        DGG = (GL - GLP) * DELTA

        ES1L = cmath.exp(S1L*Z)
        ENS1L = 1.0 / ES1L
        ES1LP = cmath.exp(S1LP*Z)
        ENS1LP = 1.0 / ES1LP
        ES3L = cmath.exp(S3L*Z)
        ENS3L = 1.0 / ES3L
        ES3LP = cmath.exp(S3LP*Z)
        ENS3LP = 1.0 / ES3LP

        FS1S1 = S1L/S1LP
        FS1S3 = S1L/S3LP
        FS3S1 = S3L/S1LP
        FS3S3 = S3L/S3LP

        A11 = A22 = DBG * (1.0 + FS1S1)
        A11 *= ES1L * ENS1LP
        A22 *= ENS1L * ES1LP
        A12 = A21 = DBG * (1.0 - FS1S1) * cmath.exp(2.*S1L*S1LP*SIGMA_L*SIGMA_L)
        A12 *= ENS1L * ENS1LP
        A21 *= ES1L * ES1LP
        A13 = A24 = DGG * (1.0 + FS3S1)
        A13 *= ES3L * ENS1LP
        A24 *= ENS3L * ES1LP
        A14 = A23 = DGG * (1.0 - FS3S1) * cmath.exp(2.*S3L*S1LP*SIGMA_L*SIGMA_L)
        A14 *= ENS3L * ENS1LP
        A23 *= ES3L * ES1LP

        A31 = A42 = DBB * (1.0 + FS1S3)
        A31 *= ES1L * ENS3LP
        A42 *= ENS1L * ES3LP
        A32 = A41 = DBB * (1.0 - FS1S3) * cmath.exp(2.*S1L*S3LP*SIGMA_L*SIGMA_L)
        A32 *= ENS1L * ENS3LP
        A41 *= ES1L * ES3LP
        A33 = A44 = DGB * (1.0 + FS3S3)
        A33 *= ES3L * ENS3LP
        A44 *= ENS3L * ES3LP
        A34 = A43 = DGB * (1.0 - FS3S3) * cmath.exp(2.*S3L*S3LP*SIGMA_L*SIGMA_L)
        A34 *= ENS3L * ENS3LP
        A43 *= ES3L * ES3LP

        #    Matrix update B=A*B
        C1 = A11*B11+A12*B21+A13*B31+A14*B41
        C2 = A21*B11+A22*B21+A23*B31+A24*B41
        C3 = A31*B11+A32*B21+A33*B31+A34*B41
        C4 = A41*B11+A42*B21+A43*B31+A44*B41
        B11 = C1
        B21 = C2
        B31 = C3
        B41 = C4

        C1 = A11*B12+A12*B22+A13*B32+A14*B42
        C2 = A21*B12+A22*B22+A23*B32+A24*B42
        C3 = A31*B12+A32*B22+A33*B32+A34*B42
        C4 = A41*B12+A42*B22+A43*B32+A44*B42
        B12 = C1
        B22 = C2
        B32 = C3
        B42 = C4

        C1 = A11*B13+A12*B23+A13*B33+A14*B43
        C2 = A21*B13+A22*B23+A23*B33+A24*B43
        C3 = A31*B13+A32*B23+A33*B33+A34*B43
        C4 = A41*B13+A42*B23+A43*B33+A44*B43
        B13 = C1
        B23 = C2
        B33 = C3
        B43 = C4

        C1 = A11*B14+A12*B24+A13*B34+A14*B44
        C2 = A21*B14+A22*B24+A23*B34+A24*B44
        C3 = A31*B14+A32*B24+A33*B34+A34*B44
        C4 = A41*B14+A42*B24+A43*B34+A44*B44
        B14 = C1
        B24 = C2
        B34 = C3
        B44 = C4

        Z += D_LP
        L = LP

    #    Done computing B = A(N)*...*A(2)*A(1)*I
    DETW = B44*B22 - B24*B42

    #    Calculate reflectivity coefficients specified by POLSTAT
    # IP = +1 fills in ++, +-, -+, --; IP = -1 only fills in -+, --.
    if POINT < MAXPOINT:
        if IP > 0:
            YA[POINT] = (B24*B41 - B21*B44)/DETW  # ++
            YB[POINT] = (B21*B42 - B41*B22)/DETW  # +-
        YC[POINT] = (B24*B43 - B23*B44)/DETW  # -+
        YD[POINT] = (B23*B42 - B43*B22)/DETW  # --

@cuda.jit(CR4XA_ALIGNED_SIG, device=True, inline=True, fastmath=True)
def Cr4xa_aligned32(N, layer_params, IP, KZ, POINT, MAXPOINT, YA, YB, YC, YD):
    EPS = np.float32(1e-10)
    CX1 = (np.float32(1.0), np.float32(0.0))
    CX0 = (np.float32(0.0), np.float32(0.0))
    PI4 = np.float32(4.0e-6 * np.pi)

    TOTAL_R = abs(KZ) <= 1e-10

    if (KZ <= -1.e-10):
        L = np.int32((N-1) * 9)
        STEP = np.int32(-9)
        SIGMA_OFFSET = -1
    else:
        L = np.int32(0)
        STEP = np.int32(9)
        SIGMA_OFFSET = 0

    #    Changing the target KZ is equivalent to subtracting the fronting
    #    medium SLD.

    # IP = 1 specifies polarization of the incident beam I+
    # IP = -1 specifies polarization of the incident beam I-
    # layer params: d, padded_sigma, rho, irho, rhoM, u1, u3

    shared_layer = cuda.shared.array(shape=(9,), dtype=numba.float32)

    tx = cuda.threadIdx.x
    if tx < 9:
        shared_layer[tx] = layer_params[L + tx]

    cuda.syncthreads()

    D_LP = shared_layer[0]
    SIGMA_LP = shared_layer[1]
    RHO_LP = shared_layer[2]
    IRHO_LP = shared_layer[3]
    RHOM_LP = shared_layer[4]
    U1_LP = (shared_layer[5], shared_layer[6])
    U3_LP = (shared_layer[7], shared_layer[8])

    # if POINT == 0:
        # print(POINT, N, L, STEP, D_LP, SIGMA_LP, RHO_LP, IRHO_LP, RHOM_LP, U1_LP.real, U1_LP.imag, U3_LP.real, U3_LP.imag)

    E0 = KZ*KZ + PI4*(RHO_LP+IP*RHOM_LP)

    Z = np.float32(0.0)

    S1LP = nsqrtcx((PI4*(RHO_LP+RHOM_LP)-E0, -PI4*(IRHO_LP+EPS)))
    S3LP = nsqrtcx((PI4*(RHO_LP-RHOM_LP)-E0, -PI4*(IRHO_LP+EPS)))
    BLP = GLP = CX0

    # if POINT == 0:
        # print(D_LP, SIGMA_LP, RHO_LP, IRHO_LP, RHOM_LP)

    if (abscx(U1_LP) <= 1.0):
        # then Bz >= 0
        # BL and GL are zero in the fronting.
        pass
    else:
        # then Bz < 0: flip!
        # This is probably impossible, since Bz defines the +z direction
        # in the fronting medium, but just in case...
        SSWAP = S1LP
        S1LP = S3LP
        S3LP = SSWAP  # swap S3 and S1

    # Initialize B matrix (identity)
    B11 = B22 = B33 = B44 = CX1
    B12 = B13 = B14 = CX0
    B21 = B23 = B24 = CX0
    B31 = B32 = B34 = CX0
    B41 = B42 = B43 = CX0

    #    Process the loop once for each interior layer, either from
    #    front to back or back to front.
    for I in range(0, N-1):
        LP = L + STEP
        S1L = S1LP  # copy from the layer before
        S3L = S3LP
        GL = GLP
        BL = BLP

        if tx < 9:
            shared_layer[tx] = layer_params[LP + tx]

        cuda.syncthreads()
        D_LP = shared_layer[0]
        SIGMA_L = shared_layer[1]
        RHO_LP = shared_layer[2]
        IRHO_LP = shared_layer[3]
        RHOM_LP = shared_layer[4]
        U1_LP = (shared_layer[5], shared_layer[6])
        U3_LP = (shared_layer[7], shared_layer[8])
        # if POINT > 991:
            # print('U3LP: ', POINT, L, tx, U3_LP[0], U3_LP[1])

        S1LP = nsqrtcx((PI4*(RHO_LP+RHOM_LP)-E0, -PI4*(IRHO_LP+EPS)))
        S3LP = nsqrtcx((PI4*(RHO_LP-RHOM_LP)-E0, -PI4*(IRHO_LP+EPS)))

        if (abscx(U1_LP) <= 1.0):
            # then Bz >= 0
            BLP = U1_LP
            GLP = invcx(U3_LP)
        else:
            # then Bz < 0: flip!
            BLP = U3_LP
            GLP = invcx(U1_LP)
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP  # swap S3 and S1

        # DELTA = 0.5 / (1.0 - (BLP*GLP))
        # DBB = (BL - BLP) * DELTA  # multiply by delta here?
        # DBG = (1.0 - BL*GLP) * DELTA
        # DGB = (1.0 - GL*BLP) * DELTA
        # DGG = (GL - GLP) * DELTA

        # ES1L = cmath.exp(S1L*Z)
        # ENS1L = 1.0 / ES1L
        # ES1LP = cmath.exp(S1LP*Z)
        # ENS1LP = 1.0 / ES1LP
        # ES3L = cmath.exp(S3L*Z)
        # ENS3L = 1.0 / ES3L
        # ES3LP = cmath.exp(S3LP*Z)
        # ENS3LP = 1.0 / ES3LP
        # if isnan(GLP[0]):
        #     print('nan GLP', N, POINT, L, U1_LP[0], U1_LP[1], U3_LP[0], U3_LP[1], RHOM_LP)
        DELTA = divcx((np.float32(0.5), np.float32(0.0)), diffcx(CX1, mulcx(BLP,GLP)))
        DBB = mulcx(diffcx(BL, BLP), DELTA)  # multiply by delta here?
        DBG = mulcx(diffcx(CX1, mulcx(BL, GLP)), DELTA)
        DGB = mulcx(diffcx(CX1, mulcx(GL, BLP)), DELTA)
        DGG = mulcx(diffcx(GL, GLP), DELTA)
        TWOSIGMA_L_SQ = (np.float32(2.0 * SIGMA_L * SIGMA_L), np.float32(0.0))
        # if isnan(DELTA[0]):
        #     print("nan DELTA")
        # if isnan(DBB[0]):
        #     print("nan DBB")
        # if isnan(DBG[0]):
        #     print("nan DBG")
        # if isnan(DGB[0]):
        #     print("nan DGB")
        # if isnan(DGG[0]):
        #     print("nan DGG")
        # ES1L = cmath.exp(S1L*Z)
        # ENS1L = 1.0 / ES1L
        # ES1LP = cmath.exp(S1LP*Z)
        # ENS1LP = 1.0 / ES1LP
        # ES3L = cmath.exp(S3L*Z)
        # ENS3L = 1.0 / ES3L
        # ES3LP = cmath.exp(S3LP*Z)
        # ENS3LP = 1.0 / ES3LP

        CXZ = (Z, np.float32(0.0))
        ES1L = expcx(mulcx(S1L, CXZ))
        ENS1L = invcx(ES1L)
        ES1LP = expcx(mulcx(S1LP, CXZ))
        ENS1LP = invcx(ES1LP)
        ES3L = expcx(mulcx(S3L, CXZ))
        ENS3L = invcx(ES3L)
        ES3LP = expcx(mulcx(S3LP, CXZ))
        ENS3LP = invcx(ES3LP)

        FS1S1 = divcx(S1L, S1LP)
        FS1S3 = divcx(S1L, S3LP)
        FS3S1 = divcx(S3L, S1LP)
        FS3S3 = divcx(S3L, S3LP)

        A11 = A22 = mulcx(DBG, sumcx(CX1, FS1S1))
        A11 = mulcx(A11, mulcx(ES1L, ENS1LP))
        A22 = mulcx(A22, mulcx(ENS1L, ES1LP))
        A12 = A21 = mulcx(mulcx(DBG, diffcx(CX1, FS1S1)), expcx(mulcx(S1L, mulcx(S1LP, TWOSIGMA_L_SQ))))
        # A12 *= ENS1L * ENS1LP
        A12 = mulcx(A12, mulcx(ENS1L, ENS1LP))
        # A21 *= ES1L * ES1LP
        A21 = mulcx(A21, mulcx(ES1L, ES1LP))
        # A13 = A24 = DGG * (1.0 + FS3S1)
        A13 = A24 = mulcx(DGG, sumcx(CX1, FS3S1))
        # A13 *= ES3L * ENS1LP
        A13 = mulcx(A13, mulcx(ES3L, ENS1LP))
        # A24 *= ENS3L * ES1LP
        A24 = mulcx(A24, mulcx(ENS3L, ES1LP))
        # A14 = A23 = DGG * (1.0 - FS3S1) * cmath.exp(2.*S3L*S1LP*SIGMA_L*SIGMA_L)
        A14 = A23 = mulcx(mulcx(DGG, diffcx(CX1, FS3S1)), expcx(mulcx(S3L, mulcx(S1LP, TWOSIGMA_L_SQ))))
        # A14 *= ENS3L * ENS1LP
        A14 = mulcx(A14, mulcx(ENS3L, ENS1LP))
        # A23 *= ES3L * ES1LP
        A23 = mulcx(A23, mulcx(ES3L, ES1LP))

        # A31 = A42 = DBB * (1.0 + FS1S3)
        A31 = A42 = mulcx(DBB, sumcx(CX1, FS1S3))
        # A31 *= ES1L * ENS3LP
        A31 = mulcx(A31, mulcx(ES1L, ENS3LP))
        # A42 *= ENS1L * ES3LP
        A42 = mulcx(A42, mulcx(ENS1L, ES3LP))
        # A32 = A41 = DBB * (1.0 - FS1S3) * cmath.exp(2.*S1L*S3LP*SIGMA_L*SIGMA_L)
        A32 = A41 = mulcx(mulcx(DBB, diffcx(CX1, FS1S3)), expcx(mulcx(S1L, mulcx(S3LP, TWOSIGMA_L_SQ))))
        # A32 *= ENS1L * ENS3LP
        A32 = mulcx(A32, mulcx(ENS1L, ENS3LP))
        # A41 *= ES1L * ES3LP
        A41 = mulcx(A41, mulcx(ES1L, ES3LP))
        # A33 = A44 = DGB * (1.0 + FS3S3)
        A33 = A44 = mulcx(DGB, sumcx(CX1, FS3S3))
        # A33 *= ES3L * ENS3LP
        A33 = mulcx(A33, mulcx(ES3L, ENS3LP))
        # A44 *= ENS3L * ES3LP
        A44 = mulcx(A44, mulcx(ENS3L, ES3LP))
        # A34 = A43 = DGB * (1.0 - FS3S3) * cmath.exp(2.*S3L*S3LP*SIGMA_L*SIGMA_L)
        A34 = A43 = mulcx(mulcx(DGB, diffcx(CX1, FS3S3)), expcx(mulcx(S3L, mulcx(S3LP, TWOSIGMA_L_SQ))))
        # A34 *= ENS3L * ENS3LP
        A34 = mulcx(A34, mulcx(ENS3L, ENS3LP))
        # A43 *= ES3L * ES3LP
        A43 = mulcx(A43, mulcx(ES3L, ES3LP))

        #    Matrix update B=A*B
        # C1= sumcx(sumcx(mulcx(A11*B11+A12, B21), mulcx(A13, B31)), sumcx(mulcx(A14, B41), mulcx($7, $8)))
        C1 = sumcx(sumcx(mulcx(A11, B11), mulcx(A12, B21)), sumcx(mulcx(A13, B31), mulcx(A14, B41)))
        C2 = sumcx(sumcx(mulcx(A21, B11), mulcx(A22, B21)), sumcx(mulcx(A23, B31), mulcx(A24, B41)))
        C3 = sumcx(sumcx(mulcx(A31, B11), mulcx(A32, B21)), sumcx(mulcx(A33, B31), mulcx(A34, B41)))
        C4 = sumcx(sumcx(mulcx(A41, B11), mulcx(A42, B21)), sumcx(mulcx(A43, B31), mulcx(A44, B41)))
        B11 = C1
        B21 = C2
        B31 = C3
        B41 = C4

        C1 = sumcx(sumcx(mulcx(A11, B12), mulcx(A12, B22)), sumcx(mulcx(A13, B32), mulcx(A14, B42)))
        C2 = sumcx(sumcx(mulcx(A21, B12), mulcx(A22, B22)), sumcx(mulcx(A23, B32), mulcx(A24, B42)))
        C3 = sumcx(sumcx(mulcx(A31, B12), mulcx(A32, B22)), sumcx(mulcx(A33, B32), mulcx(A34, B42)))
        C4 = sumcx(sumcx(mulcx(A41, B12), mulcx(A42, B22)), sumcx(mulcx(A43, B32), mulcx(A44, B42)))
        B12 = C1
        B22 = C2
        B32 = C3
        B42 = C4

        C1 = sumcx(sumcx(mulcx(A11, B13), mulcx(A12, B23)), sumcx(mulcx(A13, B33), mulcx(A14, B43)))
        C2 = sumcx(sumcx(mulcx(A21, B13), mulcx(A22, B23)), sumcx(mulcx(A23, B33), mulcx(A24, B43)))
        C3 = sumcx(sumcx(mulcx(A31, B13), mulcx(A32, B23)), sumcx(mulcx(A33, B33), mulcx(A34, B43)))
        C4 = sumcx(sumcx(mulcx(A41, B13), mulcx(A42, B23)), sumcx(mulcx(A43, B33), mulcx(A44, B43)))
        B13 = C1
        B23 = C2
        B33 = C3
        B43 = C4

        C1 = sumcx(sumcx(mulcx(A11, B14), mulcx(A12, B24)), sumcx(mulcx(A13, B34), mulcx(A14, B44)))
        C2 = sumcx(sumcx(mulcx(A21, B14), mulcx(A22, B24)), sumcx(mulcx(A23, B34), mulcx(A24, B44)))
        C3 = sumcx(sumcx(mulcx(A31, B14), mulcx(A32, B24)), sumcx(mulcx(A33, B34), mulcx(A34, B44)))
        C4 = sumcx(sumcx(mulcx(A41, B14), mulcx(A42, B24)), sumcx(mulcx(A43, B34), mulcx(A44, B44)))
        B14 = C1
        B24 = C2
        B34 = C3
        B44 = C4

        Z += D_LP
        L = LP

    #    Done computing B = A(N)*...*A(2)*A(1)*I
    DETW = diffcx(mulcx(B44, B22), mulcx(B24, B42))

    #    Calculate reflectivity coefficients specified by POLSTAT
    # IP = +1 fills in ++, +-, -+, --; IP = -1 only fills in -+, --.
    if POINT < MAXPOINT:
        if TOTAL_R:
            YA[POINT] = complex(-1., 0.0)
            YB[POINT] = complex(0.0, 0.0)
            YC[POINT] = complex(0.0, 0.0)
            YD[POINT] = complex(-1., 0.0)
        else:
            if IP > 0:
                YA[POINT] = cplx(divcx(diffcx(mulcx(B24, B41), mulcx(B21, B44)), DETW))  # ++
                YB[POINT] = cplx(divcx(diffcx(mulcx(B21, B42), mulcx(B41, B22)), DETW))  # +-
            YC[POINT] = cplx(divcx(diffcx(mulcx(B24, B43), mulcx(B23, B44)), DETW))  # -+
            YD[POINT] = cplx(divcx(diffcx(mulcx(B23, B42), mulcx(B43, B22)), DETW))  # --

        if YA[POINT] != YA[POINT]:
            print("nan: ", POINT, DETW[0], DETW[1], B44[0], B22[0], B42[0], B24[0], SIGMA_L, A34[0], A43[1])


# Cr4xa = cuda.jit(CR4XA_SIG, cache=True, device=True)(MODULE.Cr4xa)
# MODULE.Cr4xa = Cr4xa


# MAGAMP_SIG = 'void(f8[:], f8[:], f8[:], f8[:], f8[:], c16[:], c16[:], f8[:], i4[:], c16[:], c16[:], c16[:], c16[:])'
def calculate_layer_matrix(layer, D, SIGMA, IP, RHO, IRHO, RHOM, U1, U3, KZ, POINT):
    pass


MINIMAL_RHO_M = 1e-2  # in units of 1e-6/A^2

@cuda.jit(cache=True, fastmath=True)
def magnetic_amplitude_cuda(d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index, Ra, Rb, Rc, Rd):
    """
    python version of calculation
    implicit returns: Ra, Rb, Rc, Rd
    """
    #assert rho_index is None
    layers = len(d)
    points = len(KZ)

    i = cuda.grid(1)

    if i < points:
        # plus polarization must be before minus polarization because it
        # fills in all R++, R+-, R-+, R--, but minus polarization only fills
        # in R-+, R--.
        Cr4xa(layers, d, sigma, 1.0, rho, irho, rhoM, u1, u3, KZ[i], i, Ra, Rb, Rc, Rd)

        # minus polarization
        Cr4xa(layers, d, sigma, -1.0, rho, irho, rhoM, u1, u3, KZ[i], i, Ra, Rb, Rc, Rd)
    

@cuda.jit(cache=True, fastmath=False)
def magnetic_amplitude_cuda_aligned(layer_params, KZ, rho_index, Ra, Rb, Rc, Rd):
    """
    python version of calculation
    implicit returns: Ra, Rb, Rc, Rd
    """
    #assert rho_index is None
    layers = np.int32(len(layer_params) // 9)
    points = np.int32(len(KZ))

    i = cuda.grid(1)

    # if i < points:
    if True:
        # plus polarization must be before minus polarization because it
        # fills in all R++, R+-, R-+, R--, but minus polarization only fills
        # in R-+, R--.
        # Cr4xa_aligned(layers, layer_params, np.float32(1.0), KZ[i], np.int32(i), points, Ra, Rb, Rc, Rd)
        Cr4xa_aligned32(layers, layer_params, np.float32(1.0), KZ[i], np.int32(i), points, Ra, Rb, Rc, Rd)

        # minus polarization
        # Cr4xa_aligned(layers, layer_params, np.float32(-1.0), KZ[i], np.int32(i), points, Ra, Rb, Rc, Rd)
        Cr4xa_aligned32(layers, layer_params, np.float32(-1.0), KZ[i], np.int32(i), points, Ra, Rb, Rc, Rd)
    
        # print(i, Ra[i].real,)


USE_ALIGNED = True
STREAM = None

def _magnetic_amplitude(d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index, Ra, Rb, Rc, Rd):
    # print(np.abs(u1), np.abs(u3))
    # print(len(KZ))
    stream = cuda.per_thread_default_stream() if STREAM is None else STREAM
    if USE_ALIGNED:
        negative_q = KZ[0] < 0 or KZ[-1] < 0
        padded_sigma = np.zeros((sigma.shape[0] + 1,))
        if negative_q:
            padded_sigma[:-1] = sigma[:]
        else:
            padded_sigma[1:] = sigma[:]
        raw_params = [d, padded_sigma, rho, np.abs(irho), rhoM, u1.real, u1.imag, u3.real, u3.imag]
        params_32 = [p.astype(np.float32) for p in raw_params]
        # print(params_32)
        # params_forward = [p[::-1] for p in params_32] if negative_q else params_32
        # kz_forward = -KZ if negative_q else KZ
        layer_params = np.vstack(params_32).ravel('F')
        # print(layer_params)
        layer_params_d = cuda.to_device(layer_params, stream=stream)
        # print('layer params: ', layer_params[-9:])
        kz_d = cuda.to_device(KZ.astype(np.float32), stream=stream)
        rho_index_d = cuda.to_device(rho_index.astype(np.float32), stream=stream)
        # Rad, Rbd, Rcd, Rdd = [cuda.device_array(r.shape, dtype=np.complex64, stream=stream) for r in [Ra, Rb, Rc, Rd]]

    else:
        layer_params_d = [cuda.to_device(p, stream=stream) for p in (d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index)]
        kz_d = rho_index_d = None
    
    Rad, Rbd, Rcd, Rdd = [cuda.device_array(r.shape, dtype=np.complex128, stream=stream) for r in [Ra, Rb, Rc, Rd]]

    # print("stream:", stream)
    # print("default stream:", cuda.default_stream())
    # print("context:", cuda.current_context())

    threadsperblock = 16
    blockspergrid = (KZ.shape[0] + (threadsperblock - 1)) // threadsperblock
    # print(blockspergrid)
    
    # with cuda.defer_cleanup():

    if USE_ALIGNED:
        magnetic_amplitude_cuda_aligned[blockspergrid, threadsperblock, stream](layer_params_d, kz_d, rho_index_d, Rad, Rbd, Rcd, Rdd)
    else:
        magnetic_amplitude_cuda[blockspergrid, threadsperblock, stream](*layer_params_d, Rad, Rbd, Rcd, Rdd)

    return layer_params_d, kz_d, rho_index_d, Rad, Rbd, Rcd, Rdd, stream


def magnetic_amplitude(d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index, Ra, Rb, Rc, Rd):

    import time
    transfer_start = time.time()

    layer_params_d, kz_d, rho_index_d, Rad, Rbd, Rcd, Rdd, stream = _magnetic_amplitude(
        d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index, Ra, Rb, Rc, Rd
    )

    Ra[:] = Rad.copy_to_host()
    Rb[:] = Rbd.copy_to_host()
    Rc[:] = Rcd.copy_to_host()
    Rd[:] = Rdd.copy_to_host()
    
    calc_end = time.time()
    if DEBUG:
        print(f"calc time: {calc_end - transfer_start}")

    return layer_params_d, kz_d, rho_index_d, Rad, Rbd, Rcd, Rdd
