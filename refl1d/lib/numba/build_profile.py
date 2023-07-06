import numba
import numpy as np
from math import erf

SQRT1_2 = 1. / np.sqrt(2.0)

@numba.vectorize
def verf(x):
    return erf(x)

@numba.njit(cache=True)
def _build_profile(z, offset, roughness, contrast, initial_value, profiles):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points, shape = (NZ,)
    *offset*     offset for each interface, shape = (NI,)
    *roughness*  roughness of each interface, shape = (NI,)
    *contrast*   step value at each interface for each profile, shape = (NP * NI)
    *initial_value* starting value for each profile, shape = (NP)

    *profiles*   (output) results of calculation
    """
    # import time
    # start_time = time.time()
    # contrast has shape (NP, NI)
    # Number of z values:
    NZ = len(z)
    # Number of wavelength values:
    NP = initial_value.shape[0]
    # Number of interfaces:
    NI = len(offset)

    contrast_shaped = contrast.reshape((NP, NI))
    profiles_shaped = profiles.reshape((NP, NZ)) # view - updates affect profiles

    profiles_shaped += initial_value.reshape((NP, 1))
    for i in range(NI):
        offset_i = offset[i]
        sigma_i = roughness[i]
        contrast_i = contrast_shaped[:, i]
        blended = 1.0 * (z >= offset_i) if sigma_i <= 0.0 else 0.5 * verf(SQRT1_2 * (z - offset_i) / sigma_i) + 0.5
        # contrast_i has shape (NP,) and blended has shape (NZ,)
        delta = contrast_i.copy().reshape((NP, 1)) * blended.reshape((1, NZ))
        # delta has shape (NL, NZ) like result
        profiles_shaped += delta

    # end_time = time.time()
    # print(f"blend: {end_time - start_time}")
    return

def build_profile(z, offset, roughness, contrast, initial_value, profiles):
    # import time
    # start_time = time.time()
    _build_profile(z, offset, roughness, contrast, initial_value, profiles)
    # end_time = time.time()
    # print(f"build_profile: {end_time - start_time}")
    return

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
