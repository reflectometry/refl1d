import numba
import numpy as np
from .clone_module import clone_module
from .reflectivity import reflectivity_amplitude
from .magnetic import magnetic_amplitude

MODULE = clone_module("refl1d.lib.python.autosample")
MODULE.int32 = numba.int32
MODULE.float64 = numba.float64
MODULE.complex128 = numba.complex128
MODULE.boolean = numba.boolean
MODULE.reflectivity_amplitude = reflectivity_amplitude
MODULE.magnetic_amplitude = magnetic_amplitude


@numba.njit(cache=True)
def insert(arr, positions, values):
    # numba-accelerated version of numpy.insert
    ninput = len(arr)
    nvals = len(values)
    noutput = ninput + nvals
    target_positions = positions + np.arange(len(positions))
    output = np.empty((noutput,), dtype=arr.dtype)
    offsets = np.zeros((noutput,), dtype=numba.boolean)
    offsets[target_positions] = True

    input_offset = 0
    val_index = 0
    for i in numba.prange(noutput):
        if offsets[i]:
            output[i] = values[val_index]
            val_index += 1
        else:
            output[i] = arr[input_offset]
            input_offset += 1
    return output


MODULE.insert = insert

oversample_inplace = numba.njit(cache=True)(MODULE.oversample_inplace)
MODULE.oversample_inplace = oversample_inplace

get_refl_args = MODULE.get_refl_args
apply_autosampling = MODULE.apply_autosampling
autosampled_reflectivity_amplitude = MODULE.autosampled_reflectivity_amplitude

oversample_magnetic_inplace = numba.njit(cache=True)(MODULE.oversample_magnetic_inplace)
MODULE.oversample_magnetic_inplace = oversample_magnetic_inplace

autosampled_magnetic_amplitude = MODULE.autosampled_magnetic_amplitude
