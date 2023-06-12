import numpy as np
from math import erf

verf = np.vectorize(erf)

def build_profile(z, offset, roughness, value):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points
    *offset*     offset for each interface
    *roughness*  roughness of each interface
    *value*      target value for each profile and slab
    *max_rough*  limit the roughness to a fraction of the layer thickness
    """
    contrast = value[:, 1:] - value[:, :-1]
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
        blended = blend(z, sigma_i, offset_i)
        delta = contrast_i.copy().reshape((NP, 1)) * blended.reshape((1, NZ))

        # delta = contrast_i[:, None] * blended[None, :]
        result += delta
    return result

SQRT1_2 = 1. / np.sqrt(2.0)

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
