import numba
from .clone_module import clone_module

MODULE = clone_module("refl1d.lib.python.magnetic")

MODULE.prange = numba.prange

calculate_U1_U3_single = numba.njit(cache=True)(MODULE.calculate_U1_U3_single)
MODULE.calculate_U1_U3_single = calculate_U1_U3_single

calculate_u1_u3 = numba.njit(cache=True)(MODULE.calculate_u1_u3)
MODULE.calculate_u1_u3 = calculate_u1_u3


CR4XA_SIG = "void(i8, f8[:], f8[:], f8, f8[:], f8[:], f8[:], c16[:], c16[:], f8, i4, c16[:], c16[:], c16[:], c16[:])"
CR4XA_LOCALS = {
    "E0": numba.float64,
    "L": numba.int32,
    "LP": numba.int32,
    "STEP": numba.int8,
    "Z": numba.float64,
}
CR4XA_LOCALS.update(
    (s, numba.complex128)
    for s in [
        "S1L",
        "S1LP",
        "S3",
        "S3LP",
        "FS1S1",
        "FS1S3",
        "FS3S1",
        "FS3S3",
        "DELTA",
        "BL",
        "GL",
        "BLP",
        "GLP",
        "SSWAP",
        "DETW",
        "DBB",
        "DBG",
        "DGB",
        "DGG",
        "ES1L",
        "ENS1L",
        "ES1LP",
        "ENS1LP",
        "ES3L",
        "ENS3L",
        "ES3LP",
        "ENS3LP",
    ]
)
CR4XA_LOCALS.update(("A{i}{j}".format(i=i, j=j), numba.complex128) for i in range(1, 5) for j in range(1, 5))
CR4XA_LOCALS.update(("B{i}{j}".format(i=i, j=j), numba.complex128) for i in range(1, 5) for j in range(1, 5))
CR4XA_LOCALS.update(("C{i}".format(i=i), numba.complex128) for i in range(1, 5))

Cr4xa = numba.njit(CR4XA_SIG, parallel=False, cache=True, locals=CR4XA_LOCALS)(MODULE.Cr4xa)
MODULE.Cr4xa = Cr4xa


MAGAMP_SIG = "void(f8[:], f8[:], f8[:], f8[:], f8[:], c16[:], c16[:], f8[:], i4[:], c16[:], c16[:], c16[:], c16[:])"

magnetic_amplitude = numba.njit(MAGAMP_SIG, parallel=False, cache=True)(MODULE.magnetic_amplitude)
