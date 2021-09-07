from __future__ import division, print_function

import numpy as np
from numpy import inf, nan

from refl1d.lib.contract_profile import align_magnetic


# thickness, interface, rho, irho
substrate = [[nan, 10, 2, 0.2]]
air = [[nan, nan, 0, 0]]
nuclear_slope = [[2, 2, -0.2*k+5, 0] for k in range(5)]
# thickness, interface, rhoM, thetaM
magnetic_substrate = [[nan, 10, 1, 270]]
rough_magnetic_substrate = [[nan, 20, 1, 270]]
magnetic_slope = [[2, 2, -0.2*k+5, 270] for k in range(5)]
magnetic_air = [[nan, nan, 0, 270]]


def test_matched_substrate_air():
    nuclear = substrate + air
    magnetic = magnetic_substrate + magnetic_air
    expected = [
        [nan, 10, 2, 0.2, 1, 270],
        [nan, nan, 0, 0, 0, 270],
    ]
    check_one(nuclear, magnetic, expected)

def test_unmatched_substrate_air():
    nuclear = substrate + air
    magnetic = rough_magnetic_substrate + magnetic_air
    expected = [
        [nan, 10, 2, 0.2, 1, 270],
        [0, 20, 0, 0, 1, 270],
        [nan, nan, 0, 0, 0, 270],
    ]
    check_one(nuclear, magnetic, expected)

def test_offset():
    binder = [[20, 10, 4, 0.4]]
    iron = [[50, 10, 8, 0.8]]
    cap = [[10, 10, 6, 0.6]]
    magnetic_pregap = [[25, 5, 0, 270]]
    magnetic_iron = [[40, 5, 5, 270]]
    magnetic_postgap = [[15, 10, 0, 270]]
    nuclear = substrate + binder + iron + cap + air
    magnetic = (magnetic_substrate + magnetic_pregap + magnetic_iron
                + magnetic_postgap + magnetic_air)
    expected = [
        [nan, 10, 2, 0.2, 1, 270],  # substrate
        [20, 10, 4, 0.4, 0, 270],  # binder
        [5, 5, 8, 0.8, 0, 270],  # iron with dead below
        [40, 5, 8, 0.8, 5, 270],  # magnetic iron
        [5, 10, 8, 0.8, 0, 270],  # iron with dead above
        [10, 10, 6, 0.6, 0, 270],  # cap
        [nan, nan, 0, 0, 0, 270],  # air
    ]
    check_one(nuclear, magnetic, expected)


def test_stepped_nuclear():
    nuclear = substrate + nuclear_slope + air
    magnetic = magnetic_substrate + [[10, 10, 2, 270]] + magnetic_air
    expected = [
        [nan, 10, 2, 0.2, 1, 270],
        [2, 2, 5.0, 0, 2, 270],
        [2, 2, 4.8, 0, 2, 270],
        [2, 2, 4.6, 0, 2, 270],
        [2, 2, 4.4, 0, 2, 270],
        [2, 2, 4.2, 0, 2, 270],
        [0, 10, 0, 0, 2, 270],
        [nan, nan, 0, 0, 0, 270],
    ]
    check_one(nuclear, magnetic, expected)

def test_stepped_magnetic():
    nuclear = substrate + [[10, 10, 3, 0.3]] + air
    magnetic = magnetic_substrate + magnetic_slope + magnetic_air
    expected = [
        [nan, 10, 2, 0.2, 1, 270],
        [2, 2, 3, 0.3, 5.0, 270],
        [2, 2, 3, 0.3, 4.8, 270],
        [2, 2, 3, 0.3, 4.6, 270],
        [2, 2, 3, 0.3, 4.4, 270],
        [2, 10, 3, 0.3, 4.2, 270],
        [0, 2, 0, 0, 4.2, 270],
        [nan, nan, 0, 0, 0, 270],
    ]
    check_one(nuclear, magnetic, expected)

def check_one(nuclear, magnetic, expected):
    #print("nuclear", nuclear)
    #print("magnetic", magnetic)
    w, sigma, rho, irho = [np.ascontiguousarray(v, 'd') for v in zip(*nuclear)]
    wM, sigmaM, rhoM, thetaM = [np.ascontiguousarray(v, 'd') for v in zip(*magnetic)]
    result = np.empty((len(w)+len(wM), 6), 'd')
    #print("sigmaM", sigmaM)
    k = align_magnetic(w, sigma[:-1], rho, irho, wM, sigmaM[:-1], rhoM, thetaM, result)
    good = all((np.isnan(c2) and c1 == 0.) or (not np.isnan(c2) and abs(c1-c2) < 1e-10)
               for r1, r2 in zip(result[:k], expected)
               for c1, c2 in zip(r1, r2))
    if not good:
        nice = lambda v: ", ".join("["+", ".join("%g"%c for c in r)+"]" for r in v)
        raise ValueError("=== Expected:\n%s\n=== Returned:\n%s\n"
                         % (nice(expected), nice(result[:k])))

if __name__ == "__main__":
    test_matched_substrate_air()
    test_unmatched_substrate_air()
    test_stepped_nuclear()
    test_stepped_magnetic()
    test_offset()
