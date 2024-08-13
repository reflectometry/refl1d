import numba
from numba import cuda
import numpy as np

from scipy.special import erf

DEBUG = True

SQRT1_2 = numba.float32(1. / np.sqrt(2.0))

INTERFACES_PER_BLOCK = 32
MAX_NP = 4
THREADS_PER_BLOCK = INTERFACES_PER_BLOCK * MAX_NP


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
    threads_per_block = cuda.blockDim.x
    # Number of z values:
    NZ = np.int32(len(z))
    # Number of profiles to calculate:
    NP = np.int32(initial_value.shape[0])
    # Number of interfaces:
    NI = np.int32(len(offset))
    NPI = NP * NI
    # Number of interfaces per block:
    GRID_INDEX = cuda.blockIdx.x
    # thread id:
    tid = cuda.threadIdx.x
    # contrast index (0 - INTERFACES_PER_BLOCK):
    contrast_index = tid // MAX_NP
    # profile id (0 - MAX_NP):
    pid = tid % MAX_NP
    z_id = contrast_index + GRID_INDEX * INTERFACES_PER_BLOCK
    # if tid == 0:
    #     print("GRID INDEX: ", GRID_INDEX)
    #     print("z_id: ", z_id)
    
    initial_values_block = cuda.shared.array(shape=(MAX_NP,), dtype=numba.float32)
    profiles_block = cuda.shared.array(shape=(INTERFACES_PER_BLOCK, MAX_NP), dtype=numba.float32)
    contrasts_block = cuda.shared.array(shape=(INTERFACES_PER_BLOCK, MAX_NP), dtype=numba.float32)
    erf_block = cuda.shared.array(shape=(INTERFACES_PER_BLOCK, INTERFACES_PER_BLOCK), dtype=numba.float32)
    z_block = cuda.shared.array(shape=(INTERFACES_PER_BLOCK,), dtype=numba.float32)
    offsets_block = cuda.shared.array(shape=(INTERFACES_PER_BLOCK,), dtype=numba.float32)
    roughness_block = cuda.shared.array(shape=(INTERFACES_PER_BLOCK,), dtype=numba.float32)

    # set initial values for all threads
    if contrast_index == 0 and pid < NP:
        initial_values_block[pid] = initial_value[pid]
    cuda.syncthreads()

    # set sum to initial value for all threads
    if pid < NP:
        profiles_block[contrast_index, pid] = initial_values_block[pid]
    # read z values into shared memory
    if pid == 0 and z_id < NZ:
        z_block[contrast_index] = z[z_id]

    cuda.syncthreads()

    CONTRAST_OFFSET = numba.int32(0)

    # Step through all the interfaces, calculating the profile at each z value
    while CONTRAST_OFFSET < NI:
        # group memory reads:
        offset_id = contrast_index + (CONTRAST_OFFSET)
        # contrast has original shape (NP, NI), 
        contrast_id = offset_id + (pid * NI)
        if pid < NP and contrast_id < NPI:
            # read contrast values into shared memory
            contrasts_block[contrast_index, pid] = contrast[contrast_id]
        if pid == 0 and offset_id < NI:
            # read offset and roughness values into shared memory
            # when pid == 0
            offsets_block[contrast_index] = offset[offset_id]
            roughness_block[contrast_index] = roughness[offset_id]
        cuda.syncthreads()

        # calculate error function for each interface, z
        # it will be a lookup table for z_index and interface_index
        # This loop should take at most (32 // 4) = 8 iterations for magnetic,
        # and (32 // 2) = 16 iterations for nonmagnetic profiles
        ERF_Z_INDEX = contrast_index
        ERF_INTERFACE_INDEX = pid
        while (
                (ERF_INTERFACE_INDEX < INTERFACES_PER_BLOCK) and
                ((ERF_Z_INDEX + GRID_INDEX * INTERFACES_PER_BLOCK) < NZ) and 
                ((ERF_INTERFACE_INDEX + CONTRAST_OFFSET) < NI)
            ):
            offset_i = offsets_block[ERF_INTERFACE_INDEX]
            sigma_i = roughness_block[ERF_INTERFACE_INDEX]
            z_i = z_block[ERF_Z_INDEX]

            if (sigma_i <= numba.float32(0.0)):
                erf_block[ERF_Z_INDEX, ERF_INTERFACE_INDEX] = numba.float32(1.0) if z_i >= offset_i else numba.float32(0.0)
            else:
                erf_block[ERF_Z_INDEX, ERF_INTERFACE_INDEX] = numba.float32(0.5) * cuda.libdevice.erff(SQRT1_2 * (z_i - offset_i) / sigma_i) + numba.float32(0.5)
                # erf_block[ERF_Z_INDEX, ERF_INTERFACE_INDEX] = numba.float32(0.5) * erf(SQRT1_2 * (z_i - offset_i) / sigma_i) + numba.float32(0.5)
        
            ERF_INTERFACE_INDEX += MAX_NP

        cuda.syncthreads()

        # calculate profile for each interface, z
        # Now calculate the contribution from each interface at the 
        # z value for the current thread.
        if z_id < NZ and pid < NP:
            z_i = z_block[contrast_index]

            interface_index = numba.int32(0)
            max_interface_index = min(NI - CONTRAST_OFFSET, INTERFACES_PER_BLOCK)
            while interface_index < max_interface_index:
                erf_val = erf_block[contrast_index, interface_index]
                delta = contrasts_block[interface_index, pid] * erf_val
                profiles_block[contrast_index, pid] += delta
                interface_index += 1

        cuda.syncthreads()
        CONTRAST_OFFSET += INTERFACES_PER_BLOCK

    # write results back to global memory
    if z_id < NZ and pid < NP:
        profile[(NZ * pid) + z_id] = profiles_block[contrast_index, pid]


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
    from time import perf_counter

    start_time = perf_counter()
    with cuda.pinned(z, offset, roughness, contrast, initial_value, profile):
    
        stream = cuda.stream()

        z_d = cuda.to_device(z.astype(np.float32), stream=stream)
        offset_d = cuda.to_device(offset.astype(np.float32), stream=stream)
        roughness_d = cuda.to_device(roughness.astype(np.float32), stream=stream)
        contrast_d = cuda.to_device(contrast.astype(np.float32), stream=stream)
        initial_value_d = cuda.to_device(initial_value.astype(np.float32), stream=stream)
        values_transferred = perf_counter()

        NP = initial_value.shape[0]
        NZ = z.shape[0]

        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (NZ + (INTERFACES_PER_BLOCK - 1)) // INTERFACES_PER_BLOCK

        profile_d = cuda.device_array((NP * NZ,), stream=stream, dtype=np.float32)
        profile_alloc = perf_counter()

        _build_profile[blockspergrid, threadsperblock, stream](z_d, offset_d, roughness_d, contrast_d, initial_value_d, profile_d)
        kernel_time = perf_counter()
        profile[:] = profile_d.copy_to_host()[:]
        copy_time = perf_counter()
    if DEBUG:
        print("Values transferred: ", values_transferred - start_time)
        print("Profile allocation: ", profile_alloc - values_transferred)
        print("Kernel execution: ", kernel_time - profile_alloc)
        print("Copy time: ", copy_time - kernel_time)

    return

