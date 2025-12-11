import numpy as np

from refl1d.probe import abeles
from refl1d.probe.fresnel import Fresnel


def test():
    # Rough silicon with an anomolously large absorbtion
    rho, irho = 2.07, 1.01
    Vrho, Virho = -1, 1.1
    sigma = 20
    fresnel = Fresnel(rho=rho, irho=irho, Vrho=Vrho, Virho=Virho, sigma=sigma)

    Mw = [0, 0]
    Mrho = [[Vrho, rho]]
    Mirho = [[Virho, irho]]
    Msigma = [sigma]

    Q = np.linspace(-0.1, 0.1, 101, "d")
    Rf = fresnel(Q)
    rm = abeles.refl(Q / 2, depth=Mw, rho=Mrho, irho=Mirho, sigma=Msigma)
    Rm = abs(rm) ** 2

    # print "Rm", Rm
    # print "Rf", Rf
    relerr = np.linalg.norm((Rf - Rm) / Rm)
    assert relerr < 1e-14, "relative error is %g" % relerr
