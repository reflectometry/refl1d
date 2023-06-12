import numpy as np

def build_profile(z, offset, roughness, value):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points
    *offset*     offset for each interface
    *roughness*  roughness of each interface
    *value*      target value for each slab
    *max_rough*  limit the roughness to a fraction of the layer thickness
    """
    contrast = np.diff(value)
    result = np.zeros_like(z) + value[0]
    for offset_k, sigma_k, contrast_k in zip(offset, roughness, contrast):
        delta = contrast_k * blend(z, sigma_k, offset_k)
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
        return 0.5 * erf(SQRT1_2 * (z - offset) / sigma) + 0.5
