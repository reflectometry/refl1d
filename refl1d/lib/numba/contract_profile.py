import numba
from .clone_module import clone_module

MODULE = clone_module("refl1d.lib.python.contract_profile")

ALIGN_MAGNETIC_SIG = "i4(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:,:])"
# @numba.njit(ALIGN_MAGNETIC_SIG, parallel=False, cache=True)
align_magnetic = numba.njit(cache=True)(MODULE.align_magnetic)
MODULE.align_magnetic = align_magnetic


CONTRACT_MAG_SIG = "i4(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8)"
# @numba.njit(CONTRACT_MAG_SIG, parallel=False, cache=True)
contract_mag = numba.njit(cache=True)(MODULE.contract_mag)
MODULE.contract_mag = contract_mag


CONTRACT_BY_AREA_SIG = "i4(f8[:], f8[:], f8[:], f8[:], f8)"
# @numba.njit(CONTRACT_BY_AREA_SIG, parallel=False, cache=True)
contract_by_area = numba.njit(cache=True)(MODULE.contract_by_area)
MODULE.contract_by_area = contract_by_area
