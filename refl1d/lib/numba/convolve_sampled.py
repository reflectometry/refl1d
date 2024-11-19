import numba
from .clone_module import clone_module

MODULE = clone_module("refl1d.lib.python.convolve_sampled")

convolve_point_sampled = numba.njit(cache=True)(MODULE.convolve_point_sampled)
MODULE.convolve_point_sampled = convolve_point_sampled

convolve_sampled = numba.njit(cache=True)(MODULE.convolve_sampled)
