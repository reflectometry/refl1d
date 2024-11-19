import numba
from .clone_module import clone_module

MODULE = clone_module("refl1d.lib.python.reflectivity")

_REFL_SIG = "c16(i8, f8, f8[:], f8[:], f8[:], f8[:])"
_REFL_LOCALS = {
    "cutoff": numba.float64,
    "next": numba.int64,
    "sigma_offset": numba.int64,
    "step": numba.int8,
    "pi4": numba.float64,
    "kz_sq": numba.float64,
    "k": numba.complex128,
    "k_next": numba.complex128,
    "F": numba.complex128,
    "J": numba.complex128,
}
_REFL_LOCALS.update(("B{i}{j}".format(i=i, j=j), numba.complex128) for i in range(1, 3) for j in range(1, 3))
_REFL_LOCALS.update(("M{i}{j}".format(i=i, j=j), numba.complex128) for i in range(1, 3) for j in range(1, 3))
_REFL_LOCALS.update(("C{i}".format(i=i), numba.complex128) for i in range(1, 3))

refl = numba.njit(_REFL_SIG, parallel=False, cache=True, locals=_REFL_LOCALS)(MODULE.refl)
MODULE.refl = refl

REFLAMP_SIG = "void(f8[:], f8[:], f8[:,:], f8[:,:], f8[:], i4[:], c16[:])"

reflectivity_amplitude = numba.njit(REFLAMP_SIG, parallel=False, cache=True, locals={"offset": numba.int64})(
    MODULE.reflectivity_amplitude
)
