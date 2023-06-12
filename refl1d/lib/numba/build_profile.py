import numba
import numpy as np
from math import erf

SQRT1_2 = 1. / np.sqrt(2.0)

@numba.vectorize
def verf(x):
    return erf(x)

@numba.njit
def build_profile(z, offset, roughness, value):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points, shape = (NZ,)
    *offset*     offset for each interface, shape = (NI,)
    *roughness*  roughness of each interface, shape = (NI,)
    *value*      target value for each profile and slab, shape = (NP, NI + 1)
    """
    # import time
    # start_time = time.time()
    contrast = value[:, 1:] - value[:, :-1]
    # contrast has shape (NL, NI)
    # Number of z values:
    NZ = len(z)
    # Number of wavelength values:
    NP = value.shape[0]
    # Number of interfaces:
    NI = len(offset)
    initial_values = value[:, 0].copy() # contiguous
    result = np.zeros((NP, NZ)) + initial_values.reshape((NP, 1))
    for i in range(NI):
        offset_i = offset[i]
        sigma_i = roughness[i]
        contrast_i = contrast[:, i]
        blended = 1.0 * (z >= offset_i) if sigma_i <= 0.0 else 0.5 * verf(SQRT1_2 * (z - offset_i) / sigma_i) + 0.5
        # contrast_i has shape (NP,) and blended has shape (NZ,)
        delta = contrast_i.copy().reshape((NP, 1)) * blended.reshape((1, NZ))
        # delta has shape (NL, NZ) like result
        result += delta
    # end_time = time.time()
    # print(f"blend: {end_time - start_time}")
    return result


def blend(z, sigma, offset):
    """
    blend function

    Given a Gaussian roughness value, compute the portion of the neighboring
    profile you expect to find in the current profile at depth z.
    """
    if sigma <= 0.0:
        return 1.0 * (z >= offset)
    else:
        return 0.5 * erf(SQRT1_2 * (z - offset) / sigma) + 0.5
