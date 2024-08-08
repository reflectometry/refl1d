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
    *profile*    (output) results of calculation, shape = (NP * NZ,)
    """
    # import time
    # start_time = time.time()
    # contrast = value[:, 1:] - value[:, :-1]
    # contrast has shape (NL, NI)
    # Number of z values:
    NZ = np.int32(len(z))
    # Number of profiles to calculate:
    NP = np.int32(initial_value.shape[0])
    NPZ = NZ * NP
    # Number of interfaces:
    NI = np.int32(len(offset))

    # set all elements to initial_value for all profiles
    zi_offset = np.int32(0)
    for pi in range(NP):
        value = initial_value[pi]
        for zi in range(NZ):
            profile[zi_offset] = value
            zi_offset += np.int32(1)

    for i in range(NI):
        offset_i = offset[i]
        sigma_i = roughness[i]

        zi_offset = np.int32(0)
        for zi in range(NZ):
            z_i = z[zi]
            if (sigma_i <= numba.float32(0.0)):
                blended = numba.float32(1.0) if z_i >= offset_i else numba.float32(0.0)
            else:
                blended = numba.float32(0.5) * cuda.libdevice.erff(SQRT1_2 * (z_i - offset_i) / sigma_i) + numba.float32(0.5)
            
            pi_offset = i
            zi_offset = zi
            for pi in range(NP):
                contrast_i = contrast[pi_offset]
                delta = contrast_i * blended
                profile[zi_offset] += delta
                pi_offset += NI
                zi_offset += NZ


def build_profile(z, offset, roughness, contrast, initial_value, profile):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points, shape = (NZ,)
    *offset*     offset for each interface, shape = (NI,)
    *roughness*  roughness of each interface, shape = (NI,)
    *contrast*   step value at each interface for each profile, shape = (NP * NI)
    *initial_value* initial target value for each profile, shape = (NP,)
    *profile*    (output) results of calculation, shape = (NP * NZ,)
    """
    z_d = cuda.to_device(z.astype(np.float32))
    offset_d = cuda.to_device(offset.astype(np.float32))
    roughness_d = cuda.to_device(roughness.astype(np.float32))
    contrast_d = cuda.to_device(contrast.astype(np.float32))
    initial_value_d = cuda.to_device(initial_value.astype(np.float32))

    NP = initial_value.shape[0]
    NZ = z.shape[0]

    threadsperblock = 1
    blockspergrid = 1

    profile_d = cuda.device_array((NP * NZ,))
    
    _build_profile[blockspergrid, threadsperblock](z_d, offset_d, roughness_d, contrast_d, initial_value_d, profile_d)
    profile[:] = profile_d.copy_to_host()[:]
    return

