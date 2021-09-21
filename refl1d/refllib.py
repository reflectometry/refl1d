# This program is public domain
# Authors: Paul Kienzle and Brian Maranville
r"""
Reflectometry numba library
"""
__all__ = [
    "reflectivity_amplitude",
    "magnetic_amplitude",
    "calculate_u1_u3",
    "convolve_gaussian",
    "convolve_uniform",
    "convolve_sampled",
    "align_magnetic",
    "contract_by_area",
    "contract_mag",
    "rebin_counts",
    "rebin_counts_2D",
]

from .lib_numba.reflectivity import reflectivity_amplitude
from .lib_numba.magnetic import magnetic_amplitude
from .lib_numba.magnetic import calculate_u1_u3
from .lib_numba.convolve import convolve_gaussian
from .lib_numba.convolve import convolve_uniform
from .lib_numba.convolve_sampled import convolve_sampled
from .lib_numba.contract_profile import align_magnetic
from .lib_numba.contract_profile import contract_by_area
from .lib_numba.contract_profile import contract_mag
from .lib_numba.rebin import rebin_counts
from .lib_numba.rebin import rebin_counts_2D
