import numba
from .clone_module import clone_module

MODULE = clone_module("refl1d.lib.python.rebin")

# Define a bin iterator to adapt to either forward or reversed inputs.
spec = [
    ("forward", numba.boolean),
    ("n", numba.int64),
    ("edges", numba.float64[:]),
    ("bin", numba.int64),
    ("lo", numba.float64),
    ("hi", numba.float64),
    ("atend", numba.boolean),
]

BinIter = numba.experimental.jitclass(spec)(MODULE.BinIter)
MODULE.BinIter = BinIter

rebin_counts_portion = numba.njit(parallel=False, cache=True)(MODULE.rebin_counts_portion)
MODULE.rebin_counts_portion = rebin_counts_portion

rebin_counts = numba.njit(parallel=False, cache=True)(MODULE.rebin_counts)
MODULE.rebin_counts = rebin_counts

rebin_intensity = numba.njit(parallel=False, cache=True)(MODULE.rebin_intensity)

rebin_counts_2D = numba.njit(cache=True)(MODULE.rebin_counts_2D)
