import numba
import numpy as np
from math import erf

from .clone_module import clone_module

MODULE = clone_module("refl1d.lib.python.build_profile")

SQRT1_2 = 1.0 / np.sqrt(2.0)


@numba.vectorize
def verf(x):
    return erf(x)


@numba.njit(cache=True)
def blend(z, sigma, offset):
    """
    blend function

    Given a Gaussian roughness value, compute the portion of the neighboring
    profile you expect to find in the current profile at depth z.
    """
    if sigma <= 0.0:
        return 1.0 * (z >= offset)
    else:
        return 0.5 * verf(SQRT1_2 * (z - offset) / sigma) + 0.5


MODULE.blend = blend

# BUILD_PROFILE_SIG = 'void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:,:])'
# build_profile = numba.njit(BUILD_PROFILE_SIG)(MODULE.build_profile)
build_profile = numba.njit(cache=True)(MODULE.build_profile)
MODULE.build_profile = build_profile
