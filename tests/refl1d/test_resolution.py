from math import asin, degrees, log, pi, radians, sin, sqrt, tan

import numpy as np
from numpy.linalg import norm

from refl1d import instrument as inst
from refl1d import ncnrdata, snsdata
from refl1d import resolution as res


def test():
    # Test constants
    Q, dQ = 0.5, 0.01
    T, dT, L, dL = 0.5, 0.5, 2.5, 0.05
    FWHM = sqrt(log(256))
    Trad, dTrad = radians(T), radians(dT)

    # Resolution primitives
    assert norm(res.FWHM2sigma(1) - 1 / FWHM) < 1e-14
    assert norm(res.sigma2FWHM(1) - FWHM) < 1e-14
    assert norm(res.QL2T(Q=Q, L=L) - degrees(asin(Q * L / 4 / pi))) < 1e-14
    assert norm(res.TL2Q(T=T, L=L) - 4 * pi * sin(Trad) / L) < 1e-14
    assert (
        norm(
            res.dTdL2dQ(T=T, dT=dT, L=L, dL=dL)
            - 4 * pi * sin(Trad) / L * sqrt((dL / L) ** 2 + (dTrad / tan(Trad)) ** 2) / FWHM
        )
        < 1e-14
    )
    Q1 = res.TL2Q(T=T, L=L)
    dQ1 = res.dTdL2dQ(T=T, dT=dT, L=L, dL=dL)
    assert (
        norm(res.dQdT2dLoL(Q=Q1, dQ=dQ1, T=T, dT=dT) - sqrt((dQ1 * FWHM / Q1) ** 2 - (dTrad / tan(Trad)) ** 2)) < 1e-14
    )

    # For spallation sources, bin edges are at A r**n where r is 1+resolution.
    # Bin edges from 1 to 2 at 20% [1, 1.2, 1.44, 1.728, 2.0736]
    # Centers of each range          1.1  1.32  1.584  1.9008
    Lp = res.bins(1, 2, 0.2)
    dLp = res.binwidths(Lp)
    assert norm(Lp - [1.1, 1.32, 1.584, 1.9008]) < 1e-14
    assert norm(dLp - [0.2, 0.24, 0.288, 0.3456]) < 1e-14

    # Slit openings are assumed to be linear in angle; since footprint
    # goes as the sine of the angle, this is correct in the small angle
    # approximation.
    Tlo, Thi = 0.2, 0.5
    sTlo, sbelow, sabove = 0.2, 0.4, 3
    Ts = np.array([Tlo / 2, Tlo, (Tlo + Thi) / 2, Thi, Thi * 2])
    slits = (sbelow, sTlo, sTlo * (Tlo + Thi) / 2 / Tlo, sTlo * Thi / Tlo, sabove)
    assert (
        norm(
            res.slit_widths(T=Ts, slits_at_Tlo=sTlo, Tlo=Tlo, Thi=Thi, slits_below=sbelow, slits_above=sabove)[0]
            - slits
        )
        < 1e-14
    )

    # FWHM angular divergence is average slit opening / slit separation
    # For tiny samples, use the sample itself as a slit.
    d1, d2 = 3000, 1000
    s1, s2 = 0.1, 0.3
    broadening = 0.01
    d, savg = d1 - d2, (s1 + s2) / 2
    expected = degrees(savg / d)
    assert norm(res.divergence(T=T, slits=(s1, s2), distance=(d1, d2)) - expected) < 1e-14
    assert (
        norm(
            res.divergence(T=T, slits=(s1, s2), distance=(d1, d2), sample_broadening=broadening)
            - (expected + broadening)
        )
        < 1e-14
    )
    assert (
        norm(
            res.divergence(T=T, slits=(s1, s2), distance=(d1, d2), sample_width=1)
            - degrees((s1 + sin(Trad)) / (2 * d1))
        )
        < 1e-14
    )

    # Simulate an scanning reflectometer
    mono = inst.Monochromatic(
        d_s1=d1,
        d_s2=d2,
        wavelength=L,
        dLoL=dL / L,
        slits_at_Tlo=sTlo,
        Tlo=Tlo,
        Thi=Thi,
        slits_below=sbelow,
        slits_above=sabove,
    )
    sim = mono.probe(T=Ts, sample_broadening=broadening, sample_width=10)

    # Check the result
    def mono_dTdQ():
        slits = res.slit_widths(T=Ts, slits_at_Tlo=sTlo, Tlo=Tlo, Thi=Thi, slits_below=sbelow, slits_above=sabove)
        dT = res.divergence(T=Ts, slits=slits, distance=(d1, d2), sample_broadening=broadening, sample_width=10)
        dQ = res.dTdL2dQ(T=Ts, dT=dT, L=L, dL=dL)
        return dT, dQ

    e_dT, e_dQ = mono_dTdQ()
    assert norm(sim.T - Ts) < 1e-13, str(norm(sim.T - Ts))
    assert norm(sim.dT + broadening - e_dT) < 1e-13, str(norm(sim.dT + broadening - e_dT))
    assert norm(sim.L - L) < 1e-13, str(norm(sim.L - L))
    assert norm(sim.dL - dL) < 1e-13, str(norm(sim.dL - dL))
    assert norm(sim.Q - res.TL2Q(Ts, L)) < 1e-13, str(norm(sim.Q - res.TL2Q(Ts, L)))
    assert norm(sim.dQ - e_dQ) < 1e-13, str(norm(sim.dQ - e_dQ))
    assert isinstance(str(mono), type(""))
    assert isinstance(ncnrdata.NG1.defaults(), type(""))

    # Check the subclassing interface
    class Mono(inst.Monochromatic):
        instrument = "mono"
        d_s1, d_s2 = d1, d2
        wavelength = L
        dLoL = dL / L

    # Set some default parameters
    mono2 = Mono(slits_at_Tlo=sTlo, Tlo=Tlo, Thi=Thi, slits_below=sbelow, slits_above=sabove)
    sim = mono2.probe(T=Ts, sample_broadening=0.01, sample_width=10)
    assert norm(sim.dQ - e_dQ) < 1e-14

    # Simulate an TOF reflectometer
    poly = inst.Pulsed(d_s1=d1, d_s2=d2, wavelength=(1, 2), dLoL=0.2)
    sim = poly.probe(T=T, slits=(s1, s2), sample_broadening=broadening, sample_width=10)

    # Check the result
    def poly_dTdQ():
        low, high = poly.wavelength
        L = res.bins(low, high, poly.dLoL)[::-1]  # reversed to be in Q order
        dL = res.binwidths(L)
        slits = (s1, s2)
        dT = res.divergence(T=T, slits=slits, distance=(d1, d2), sample_broadening=broadening, sample_width=10)
        dQ = res.dTdL2dQ(T=T, dT=dT, L=L, dL=dL)
        return dT, dQ, L, dL

    e_dT, e_dQ, Ls, e_dL = poly_dTdQ()
    assert norm(sim.T - T) < 1e-14, str(norm(sim.T - T))
    assert norm(sim.dT + broadening - e_dT) < 1e-14, str(norm(sim.dT + broadening - e_dT))
    assert norm(sim.L - Ls) < 1e-14, str(norm(sim.L - Ls))
    assert norm(sim.dL - e_dL) < 1e-14, str(norm(sim.dL - e_dL))
    assert norm(sim.Q - res.TL2Q(T, Ls)) < 1e-14, str(norm(sim.Q - res.TL2Q(T, Ls)))
    assert norm(sim.dQ - e_dQ) < 1e-14, str(norm(sim.dQ - e_dQ))

    assert isinstance(str(poly), type(""))
    assert isinstance(snsdata.Liquids.defaults(), type(""))

    # Make sure that the string reps don't crash
    _ = (str(mono), str(ncnrdata.NG1.defaults()), str(ncnrdata.NG1()), str(poly), str(snsdata.Liquids.defaults()))
    if 1:
        for s in _:
            print(s)


if __name__ == "__main__":
    test()
