import os
from os.path import join as joinpath, dirname, exists, getmtime as filetime
import tempfile

from bumps.util import pushdir
import matplotlib.pyplot as plt
import numpy as np
from numpy import radians

from refl1d.models.sample.reflectivity import magnetic_amplitude as refl

H2K = 2.91451e-5
B2SLD = 2.31929e-06
GEPORE_SRC = "gepore_zeeman.f"


def add_H(layers, H=0.0, AGUIDE=270.0):
    """Take H (vector) as input and add H to 4piM:"""
    new_layers = []
    for layer in layers:
        thickness, sld_n, sld_m, theta_m, phi_m = layer
        # we read phi_m, but it must be zero so we don't use it.
        sld_m_x = sld_m * np.cos(theta_m * np.pi / 180.0)  # phi_m = 0
        sld_m_y = sld_m * np.sin(theta_m * np.pi / 180.0)  # phi_m = 0
        sld_m_z = 0.0  # by Maxwell's equations, H_demag = mz so we'll just cancel it here
        sld_h = B2SLD * 1.0e6 * H
        # this code was completely wrong except for the case AGUIDE=270
        sld_h_x = 0  # by definition, H is along the z,lab direction and x,lab = x,sam so Hx,sam must = 0
        sld_h_y = -sld_h * np.sin(AGUIDE * np.pi / 180.0)
        sld_h_z = sld_h * np.cos(AGUIDE * np.pi / 180.0)
        sld_b_x = sld_h_x + sld_m_x
        sld_b_y = sld_h_y + sld_m_y
        sld_b_z = sld_h_z + sld_m_z
        sld_b = np.sqrt((sld_b_z) ** 2 + (sld_b_y) ** 2 + (sld_b_x) ** 2)
        # this was wrong:
        # theta_b = np.arctan2(sld_b_y, sld_b_x)
        theta_b = np.arccos(sld_b_x / sld_b)
        # this didn't hurt anything but is also unneeded:
        # theta_b = np.mod(theta_b, 2.0*np.pi)
        # this wasn't even close to correct:
        # phi_b = np.arcsin(sld_b_z/sld_b)
        phi_b = np.arctan2(sld_b_z, sld_b_y)
        phi_b = np.mod(phi_b, 2.0 * np.pi)
        new_layer = [thickness, sld_n, sld_b, theta_b * 180.0 / np.pi, phi_b * 180.0 / np.pi]
        new_layers.append(new_layer)
    return new_layers


def gepore(layers, QS, DQ, NQ, EPS, H):
    # if H != 0:
    layers = add_H(layers, H, AGUIDE=EPS)
    # layers = add_H(layers, H, EPS-270, 0)
    depth, rho, rhoB, thetaB, phiB = list(zip(*layers))

    NL = len(rho) - 2
    NC = 1
    ROSUP = rho[-1] + rhoB[-1]
    ROSUM = rho[-1] - rhoB[-1]
    ROINP = rho[0] + rhoB[0]
    ROINM = rho[0] - rhoB[0]

    path = tempfile.gettempdir()
    header = joinpath(path, "inpt.d")
    layers = joinpath(path, "tro.d")
    rm_real = joinpath(path, "rrem.d")
    rm_imag = joinpath(path, "rimm.d")
    rp_real = joinpath(path, "rrep.d")
    rp_imag = joinpath(path, "rimp.d")

    # recompile gepore if necessary
    gepore = joinpath(path, "gepore")
    gepore_source = joinpath(dirname(__file__), "..", "..", "refl1d", "lib", GEPORE_SRC)
    if not exists(gepore) or filetime(gepore) < filetime(gepore_source):
        status = os.system("gfortran -O2 -o %s %s" % (gepore, gepore_source))
        if status != 0:
            raise RuntimeError("Could not compile %r" % gepore_source)
        if not exists(gepore):
            raise RuntimeError("No gepore created in %r" % gepore)

    with open(layers, "w") as fid:
        for T, BN, PN, THE, PHI in list(zip(depth, rho, rhoB, thetaB, phiB))[1:-1]:
            fid.write("%f %e %e %f %f\n" % (T, 1e-6 * BN, 1e-6 * PN, radians(THE), radians(PHI)))

    for IP in (0.0, 1.0):
        with open(header, "w") as fid:
            fid.write(
                "%d %d %f %f %d %f (%f,0.0) (%f,0.0) %e %e %e %e\n"
                % (NL, NC, QS, DQ, NQ, radians(EPS), IP, 1 - IP, 1e-6 * ROINP, 1e-6 * ROINM, 1e-6 * ROSUP, 1e-6 * ROSUM)
            )
        with pushdir(path):
            status = os.system("./gepore")  # >/dev/null')
            if status != 0:
                raise RuntimeError("Could not run gepore")
        rp = np.loadtxt(rp_real).T[1] + 1j * np.loadtxt(rp_imag).T[1]
        rm = np.loadtxt(rm_real).T[1] + 1j * np.loadtxt(rm_imag).T[1]
        if IP == 1.0:
            Rpp, Rpm = rp, rm
        else:
            Rmp, Rmm = rp, rm
    return Rmm, Rpm, Rmp, Rpp


def magnetic_cc(layers, kz, Aguide, H):
    depth, rho, rhoM, thetaM, phiM = list(zip(*layers))
    R = refl(kz, depth, rho, 0, rhoM, thetaM, 0, Aguide, H)
    return R


def Rplot(title, Qz, R, format, dataset):
    """plot reflectivity"""
    # plt.hold(True)
    for name, xs in zip(("++", "+-", "-+", "--"), R):
        # for name,xs in zip(('--','-+','+-','++'),R):
        Rxs = abs(xs) ** 2
        if (Rxs > 1e-8).any():
            plt.plot(Qz, Rxs, format, label=name + dataset)
    plt.xlabel("$2k_{z0}$", size="large")
    plt.ylabel("R")
    plt.legend()
    plt.yscale("log")
    plt.title(title)


def rplot(Qz, R, format):
    """plot real and imaginary"""
    # plt.hold(True)
    plt.figure()
    for name, xs in zip(("++", "+-", "-+", "--"), R):
        rr = xs.real
        if (rr > 1e-8).any():
            plt.plot(Qz, rr, format, label=name + "r")
    plt.legend()
    plt.figure()
    for name, xs in zip(("++", "+-", "-+", "--"), R):
        ri = xs.imag
        if (ri > 1e-8).any():
            plt.plot(Qz, ri, format, label=name + "i")
    plt.legend()

    plt.figure()
    for name, xs in zip(("++", "+-", "-+", "--"), R):
        phi = np.arctan2(xs.imag, xs.real)
        if (ri > 1e-8).any():
            plt.plot(Qz, phi, format, label=name + "i")
    plt.legend()


def profile_plot(layers):
    dz, rho, rhoM, thetaM, phiM = np.asarray(layers).T
    z = np.cumsum([np.hstack((-dz[0], dz))])
    rho, rhoM, thetaM = [np.hstack((v[0], v)) for v in (rho, rhoM, thetaM)]
    plt.step(z, rho, label="rho")
    plt.step(z, rhoM, label="rhoM")
    plt.step(z, thetaM * 2 * np.pi / 360.0, label="thetaM")
    plt.legend()


def compare(name, layers, Aguide=270, H=0):
    QS = 0.001
    DQ = 0.0002
    NQ = 300
    Rgepore = gepore(layers, QS, DQ, NQ, Aguide, H)

    Qz = np.arange(NQ) * DQ + QS
    kz = Qz / 2
    Rrefl1d = magnetic_cc(layers, kz, Aguide, H)

    plt.subplot(211)
    title = "%s H=%g" % (name, H)
    Rplot(title, Qz, Rgepore, "-", "gepore")
    Rplot(title, 2 * kz, Rrefl1d, "-.", "refl1d")

    plt.subplot(212)
    profile_plot(layers)

    return kz, Rrefl1d

    assert np.linalg.norm((R[0] - Rpp) / Rpp) < 1e-13, "fail ++ %s" % name
    assert np.linalg.norm((R[1] - Rpm) / Rpm) < 1e-13, "fail +- %s" % name
    assert np.linalg.norm((R[2] - Rmp) / Rmp) < 1e-13, "fail -+ %s" % name
    assert np.linalg.norm((R[3] - Rmm) / Rmm) < 1e-13, "fail -- %s" % name


def simple():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 270, 0],
        [200, 4.0, 1.0, 359.9, 0.0],
        [200, 2.0, 1.0, 270, 0.0],
        [0, 4.0, 0.0, 270, 0.0],
    ]
    return "Si-Fe-Au-Air", layers, Aguide


def twist():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 2.1, 0.0, 270, 0.0],
        [20, 8.0, 5.0, 270, 0.0],
        [20, 8.0, 5.0, 220, 0.0],
        [20, 8.0, 5.0, 180, 0.0],
        [10, 4.5, 0.0, 270, 0.0],
        [0, 0.0, 0.0, 270, 0.0],
    ]
    return "twist", layers, Aguide


def magsub():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [50, 8.0, 5.0, 270, 0.0],
        [0, 2.1, 0.0, 270, 0.0],
        [10, 4.5, 0.0, 270, 0.0],
        [0, 0.0, 0.0, 270, 0.0],
    ]
    return "magnetic substrate", layers, Aguide


def magsurf():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 270, 0.0],
        [200, 4.0, 1.0, 0.01, 0.0],
        [200, 2.0, 1.0, 270, 0.0],
        [200, 4.0, 0.0, 270, 0.0],
    ]
    return "magnetic surface", layers, Aguide


def NSF_example():
    Aguide = 270.0
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 1e-6, 90, 0.0],
        [200, 4.0, 1.0, 90, 0.0],
        [200, 2.0, 1.0, 90, 0.0],
        [200, 4.0, 1e-6, 90, 0.0],
    ]
    return "non spin flip", layers, Aguide


def Yaohua_example():
    Aguide = 270.0
    rhoB = B2SLD * 0.4 * 1e6
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, rhoB, 90, 0.0],
        [200, 4.0, rhoB + 1.0, np.degrees(np.arctan2(rhoB, 1.0)), 0.0],
        [200, 2.0, rhoB + 1.0, 90, 0.0],
        [0, 4.0, rhoB, 90, 0.0],
    ]
    return "Yaohua example", layers, Aguide


def zf_Yaohua_example():
    Aguide = 270.0
    layers = [
        # depth rho rhoM thetaM  phiM
        [100, 0.0, 0.0, 90, 0.0],
        [200, 4.0, 1.0, 0.0001, 0.0],
        [323, 2.0, 2.0, 90, 0.0],
        [100, 4.0, 0.0, 90, 0.0],
    ]
    return "Yaohua example zf", layers, Aguide


def Chuck_example():
    Aguide = 270.0
    layers = [
        # depth rho rhoM thetaM    phiM
        [0, 2.0, 2.0, 90.0, 0.0],
        [200, 6.0, 4.0, 0.0001, 0.0],
        [300, 6.0, 4.0, 0.0001, 0.0],
        [0, 5.0, 0.0001, 90, 0.0],
    ]
    return "Chuck example", layers, Aguide


def Kirby_example():
    Aguide = 0  # 5
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 90, 0.0],
        [50, 4.0, 0.0, 0.001, 0.0],
        [825, 9.06, 1.12, 0.0001, 0.0],
        [0, 2.07, 0.0001, 90, 0.0],
    ]
    return "Kirby example", layers, Aguide


def zeeman_paper_example():
    Aguide = 6.52298771199324
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 270.0, 0.0],
        [219.2, 4.109, 1.334e-5, 270.0, 0.0],
        [672.26, 7.991, 1.121, 270.0, 0.0],
        [0, 2.07, 0.0, 270.0, 0.0],
    ]
    return "zeeman paper", layers, Aguide


def _random_model(Nlayers=10, seed=None):
    if seed is None:
        seed = np.random.randint(1000000)
    np.random.seed(seed)
    dz = np.random.rand(Nlayers) * 2000.0 / Nlayers
    dz[0] = dz[-1] = 100.0
    rho = 8 * (np.random.rand(Nlayers) - 0.1)
    rhoM = 2 * np.random.rand(Nlayers)
    thetaM = 90.0 * np.random.rand(Nlayers)
    thetaM[-1] = thetaM[0] = 90.0
    rhoM[0] = rhoM[-1] = 0.0
    phiM = np.zeros_like(dz)
    layers = np.vstack((dz, rho, rhoM, thetaM, phiM)).T.tolist()
    Aguide = np.random.rand() * 360.0
    # Aguide = 273.
    return "random(%d)" % seed, layers, Aguide


def demo():
    """run demo"""
    # plt.figure(); compare(*simple())
    # plt.figure(); compare(*twist())
    # plt.figure(); compare(*magsub())
    # plt.figure(); compare(*magsurf())
    plt.figure()
    compare(*zf_Yaohua_example(), H=0.4)  # 4000 gauss
    plt.figure()
    compare(*zf_Yaohua_example(), H=0.0005)  # 5 Gauss
    plt.figure()
    compare(*Kirby_example(), H=0.244)
    # plt.figure(); compare(*NSF_example(), H=0.00005) # Earth's field, 0.5G
    # plt.figure(); compare(*NSF_example(), H=1.0) # 1 tesla
    # plt.figure(); compare(*Chuck_example(), H=0) # zeroish field, but magnetic front
    # plt.figure(); compare(*_random_model(), H=0)
    plt.figure()
    compare(*_random_model(), H=np.random.rand() * 2)
    # plt.figure(); compare(*_random_model(seed=998543), H=2)
    plt.show()


def write_Chuck_result():
    kz, R = compare(*Chuck_example(), H=0)  # zeroish field, but magnetic front
    np.savetxt("Rmm.txt", np.vstack((2 * kz, np.abs(R[3]) ** 2)).T, delimiter="\t")
    np.savetxt("Rmp.txt", np.vstack((2 * kz, np.abs(R[2]) ** 2)).T, delimiter="\t")
    np.savetxt("Rpm.txt", np.vstack((2 * kz, np.abs(R[1]) ** 2)).T, delimiter="\t")
    np.savetxt("Rpp.txt", np.vstack((2 * kz, np.abs(R[0]) ** 2)).T, delimiter="\t")


if __name__ == "__main__":
    demo()
