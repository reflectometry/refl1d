import numba
from numba import cuda
import numpy as np

SQRT1_2 = numba.float32(1. / np.sqrt(2.0))

@cuda.jit
def _build_profile(z, offset, roughness, contrast, initial_value, profile):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points, shape = (NZ,)
    *offset*     offset for each interface, shape = (NI,)
    *roughness*  roughness of each interface, shape = (NI,)
    *contrast*   difference in target value for each profile and slab, shape = (NP, NI)
    *initial_value* initial target value for each profile, shape = (NP,)
    """
    # import time
    # start_time = time.time()
    # contrast = value[:, 1:] - value[:, :-1]
    # contrast has shape (NL, NI)
    # Number of z values:
    NZ = len(z)
    # Number of profiles to calculate:
    NP = contrast.shape[0]
    NPZ = NZ * NP
    # Number of interfaces:
    NI = len(offset)

    # set all elements to initial_value for all profiles
    for pi in range(NP * NZ, dtype=np.int32):
        value = initial_value[pi]
        for zi in range(pi, NPZ, NP, dtype=np.int32):
            profile[zi] = value

    pi_offset = np.int32(0)
    for i in range(NI, dtype=np.int32):
        offset_i = offset[i]
        sigma_i = roughness[i]

        zi_offset = np.int32(0)
        for zi in range(NZ, dtype=np.int32):
            z_i = z[zi]
            if (sigma_i <= numba.float32(0.0)):
                blended = numba.float32(1.0) if z_i >= offset_i else numba.float32(0.0)
            else:
                blended = numba.float32(0.5) * cuda.libdevice.erff(SQRT1_2 * (z_i - offset_i) / sigma_i) + numba.float32(0.5)
            
            for pi in range(NP, dtype=np.int32):
                contrast_i = contrast[pi_offset + pi]
                delta = contrast_i * blended
                profile[zi_offset] += delta
                zi_offset += np.int32(1)
        pi_offset += NP


def build_profile(z, offset, roughness, value):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points, shape = (NZ,)
    *offset*     offset for each interface, shape = (NI,)
    *roughness*  roughness of each interface, shape = (NI,)
    *value*      target value for each slab, shape = (NP, NI + 1)
    *initial_value* initial target value for each profile, shape = (NP,)
    """
    z_d = cuda.to_device(z.astype(np.float32))
    offset_d = cuda.to_device(offset.astype(np.float32))
    roughness_d = cuda.to_device(roughness.astype(np.float32))
    contrast = value[:, 1:] - value[:, :-1]
    initial_value = value[:, 0]
    contrast_d = cuda.to_device(contrast.astype(np.float32))
    initial_value_d = cuda.to_device(initial_value.astype(np.float32))

    NP = contrast.shape[0]
    NZ = z.shape[0]

    threadsperblock = 1
    blockspergrid = 1

    profile_d = cuda.device_array((NP * NZ,))
    
    _build_profile[blockspergrid, threadsperblock](z_d, offset_d, roughness_d, contrast_d, initial_value_d, profile_d)
    profile = profile_d.copy_to_host()
    profile = profile.reshape((NP, NZ))
    return profile

