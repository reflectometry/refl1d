import numpy as np
from numpy import sqrt
from numpy.random import randn

from refl1d.models import Experiment, Material, Probe, air, silicon, D2O, SNS as snsdata

Probe.view = "log"  # log, linear, fresnel, or Q**4


def gendata(sample, filename, T, slits=None, dLoL=0.03, counts=None):
    if slits is None:
        slits = (0.2 * T, 0.2 * T)
    if counts is None:
        counts = (70 * T) ** 4
    # Compute reflectivity with resolution and added noise
    instrument = snsdata.Liquids()
    probe = instrument.simulate(T=T, slits=slits, dLoL=dLoL)
    M = Experiment(probe=probe, sample=sample)
    Q, R = M.reflectivity()
    I = snsdata.feather(probe.L, counts=counts)
    dR = sqrt(R / I)
    R += randn(len(Q)) * dR
    probe.R, probe.dR = R, dR
    probe.I = I  # Needed for stitching

    # Save to file
    data = np.array((probe.Q, probe.dQ, probe.R, probe.dR, probe.L))
    header = """\
#F /SNSlocal/REF_L/2007_1_4B_SCI/2893/NeXus/REF_L_1001.nxs
#E 1174593179.87
#D 2007-03-22 15:52:59
#C Run Number: 1001
#C Title: %(sample)s
#C Notes: Fake data for %(sample)s
#C Detector Angle: (%(angle)g, 'degree')
#C Proton Charge: 35.6334991455

#S 1 Spectrum ID ('bank1', (87, 152))
#N 3
#L Q(inv Angstrom) dQ(inv Angstrom) R() dR() L(Angstrom)
""" % dict(angle=T, sample=str(sample))
    outfile = open(filename, "w")
    outfile.write(header)
    np.savetxt(outfile, data.T)
    outfile.close()

    return probe


def main():
    # Simulate a sample
    SiO2 = Material("SiO2", density=2.634)
    # sample1 = silicon%1 + SiO2/200%2 + air
    # sample2 = silicon%1 + SiO2/200%2 + D2O
    sample1 = air % 2 + SiO2 / 200 % 1 + silicon
    sample2 = D2O % 2 + SiO2 / 200 % 1 + silicon

    # Measurement parameters
    T = 1.0
    slits = (0.2, 0.2)
    dLoL = 0.03

    sets = []
    for T in [0.7, 4]:
        s = 0.2 * T
        n = (70 * T) ** 4
        p1 = gendata(sample1, "surround_air_%g.txt" % T, T=T, slits=(s, s), counts=n)
        p2 = gendata(sample2, "surround_d2o_%g.txt" % T, T=T, slits=(s, s), counts=n)
        sets.append((p1, p2))
    #    p1.plot(); p2.plot()
    # stitch_sets(sets)
    # import matplotlib.pyplot as plt; plt.show()


def plot_log(data, theory=None):
    import matplotlib.pyplot as plt

    Q, dQ, R, dR = data
    # plt.errorbar(Q, R, yerr=dR, xerr=dQ, fmt='.')
    plt.plot(Q, R, "-x", hold=True)
    if theory is not None:
        Q, R = theory
        plt.plot(Q, R, hold=True)
    plt.yscale("log")
    plt.xlabel("Q (inv Angstroms)")
    plt.ylabel("Reflectivity")


# TODO: move stitch from an example to a test suite
def stitch_sets(sets):
    from refl1d.models.probe import stitch

    p1, p2 = zip(*sets)
    plot_log(stitch(p1, same_Q=0.001))
    plot_log(stitch(p2, same_Q=0.001))


if __name__ == "__main__":
    main()
