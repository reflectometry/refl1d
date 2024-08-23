import numba
from .clone_module import clone_module

MODULE = clone_module("refl1d.lib.python.convolve")

MODULE.prange = numba.prange

convolve_uniform = numba.njit("(f8[:], f8[:], f8[:], f8[:], f8[:])", cache=True, parallel=False)(
    MODULE.convolve_uniform
)

convolve_gaussian_point = numba.njit(
    "f8(f8[:], f8[:], i8, i8, f8, f8, f8)",
    cache=True,
    parallel=False,
    locals={
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
    },
)(MODULE.convolve_gaussian_point)
MODULE.convolve_gaussian_point = convolve_gaussian_point

# has same performance when using guvectorize instead of njit:
# @numba.guvectorize("(i8, f8[:], f8[:], i8, f8[:], f8[:], f8[:])", '(),(m),(m),(),(n),(n)->(n)')

convolve_gaussian = numba.njit(
    "(f8[:], f8[:], f8[:], f8[:], f8[:])",
    cache=True,
    parallel=False,
    locals={
        "sigma": numba.float64,
        "xo": numba.float64,
        "limit": numba.float64,
        "k_in": numba.int64,
        "k_out": numba.int64,
    },
)(MODULE.convolve_gaussian)
