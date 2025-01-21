import functools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy import radians

from refl1d.sample.reflectivity import magnetic_amplitude as magrefl, reflectivity_amplitude as refl
from gepore_runner import GeporeRunner

B2SLD = 2.31929e-06
GEPORE_SRC = "gepore.f"
GEPORE_ZEEMAN_SRC = "gepore_zeeman.f"

XS_LABELS = ["++", "+-", "-+", "--"]


def magnetic_cc(layers, kz, Aguide, H):
    depth, rho, rhoM, thetaM, phiM = list(zip(*layers))
    R = magrefl(kz, depth, rho, 0, rhoM, thetaM, 0, Aguide, H)
    # magnetic_amplitude returns cross-sections in order --, -+, +-, ++
    # so we need to reverse them here to compare to gepore outputs
    return R[::-1]


def nsf_refl(layers, kz, Aguide, H):
    depth, rho, rhoM, thetaM, phiM = list(zip(*layers))
    rho_plus = rho - rhoM * np.sin(radians(thetaM))  # in our convention, thetaM == 270 means aligned with the field
    rho_minus = rho + rhoM * np.sin(radians(thetaM))
    R_plus = refl(kz, depth, rho_plus, 0, 0, None)
    R_minus = refl(kz, depth, rho_minus, 0, 0, None)
    sf = np.zeros_like(R_plus)
    return R_plus, sf, sf, R_minus


def Rplot(axes, Qz, R, format, dataset, **plot_kw):
    """plot reflectivity"""
    # plt.hold(True)
    axes.set_prop_cycle(None)
    for name, xs in zip(XS_LABELS, R):
        Rxs = xs * xs.conj()
        if (Rxs > 1e-8).any():
            axes.plot(Qz, Rxs, format, label=name + dataset, **plot_kw)


def rplot(Qz, R, format):
    """plot real and imaginary"""
    # plt.hold(True)
    plt.figure()
    for name, xs in zip(XS_LABELS, R):
        rr = xs.real
        if (rr > 1e-8).any():
            plt.plot(Qz, rr, format, label=name + "r")
    plt.legend()
    plt.figure()
    for name, xs in zip(XS_LABELS, R):
        ri = xs.imag
        if (ri > 1e-8).any():
            plt.plot(Qz, ri, format, label=name + "i")
    plt.legend()

    plt.figure()
    for name, xs in zip(XS_LABELS, R):
        phi = np.arctan2(xs.imag, xs.real)
        if (ri > 1e-8).any():
            plt.plot(Qz, phi, format, label=name + "i")
    plt.legend()


def profile_plot(axes: plt.Axes, layers):
    dz, rho, rhoM, thetaM, phiM = np.asarray(layers).T
    z = np.cumsum([np.hstack((-dz[0], dz))])
    rho, rhoM, thetaM = [np.hstack((v[0], v)) for v in (rho, rhoM, thetaM)]
    axes.step(z, rho, label="rho")
    axes.step(z, rhoM, label="rhoM")
    axes.step(z, thetaM * 2 * np.pi / 360.0, label="thetaM")
    axes.legend()


def run_comparison(
    gepore_runner, name, layers, Aguide=270, H=0, zeeman_corrections=True, nsf_compare=False, save_output=True
):
    if save_output:
        folder_name = f"{name.replace(' ', '_')}_H{H:.3f}_A{Aguide:.3f}_z{zeeman_corrections}"
        save_folder = (Path("output") / folder_name).absolute()
        save_folder.mkdir(exist_ok=True, parents=True)
    else:
        save_folder = None

    figure = plt.figure()
    axes = figure.subplots(2, 1)
    figure.tight_layout()
    ax_refl: plt.Axes = axes[0]
    ax_profile: plt.Axes = axes[1]

    QS = 0.001
    DQ = 0.0004
    NQ = 80
    EPS = -Aguide
    Rgepore = gepore_runner.run(
        layers, QS, DQ, NQ, EPS, H, zeeman_corrections=zeeman_corrections, output_folder=save_folder
    )

    Qz = np.arange(NQ) * DQ + QS
    kz = Qz / 2
    Rrefl1d = magnetic_cc(layers, kz, Aguide, H)

    title = "%s H=%g" % (name, H)
    Rplot(ax_refl, Qz, Rgepore, "-", "gepore")
    Rplot(ax_refl, 2 * kz, Rrefl1d, "o", "refl1d", markevery=1, fillstyle="none", markersize=3)
    if nsf_compare:
        Rnsf = nsf_refl(layers, kz, Aguide, H)
        Rplot(ax_refl, 2 * kz, Rnsf, "+", "nsf", markevery=1, fillstyle="none", markersize=3)
    ax_refl.set_title(title)
    ax_refl.set_xlabel("$2k_{z0}$", size="large")
    ax_refl.set_ylabel("R")
    ax_refl.set_yscale("log")
    ax_refl.legend()

    profile_plot(ax_profile, layers)

    if save_folder is not None:
        plt.savefig(save_folder / "comparison.png")
        refl1d_output = np.vstack((2 * kz, *Rrefl1d)).T
        print(refl1d_output.shape)
        np.savetxt(save_folder / "refl1d.dat", refl1d_output, delimiter="\t", header="2kz\tRpp\tRpm\tRmp\tRmm")

    Rpp, Rpm, Rmp, Rmm = Rgepore
    for index, (name, Rg, R1) in enumerate(zip(XS_LABELS, Rgepore, Rrefl1d)):
        if np.any(R1 > 1e-8):
            mask = np.logical_and(R1 > 1e-8, np.isfinite(R1))
            norm_diff_real = (Rg.real[mask] - R1.real[mask]) / R1.real[mask]
            max_diff = np.linalg.norm(norm_diff_real)
            # max_diff = np.max(np.abs(norm_diff_real[np.isfinite(norm_diff_real)]))
            # assert max_diff < 1e-6
            print(f"{name} gepore vs refl1d magnetic: {max_diff:.2e}")
        if nsf_compare:
            Rn = Rnsf[index]
            if np.any(Rn > 1e-8):
                mask = np.logical_and(Rn > 1e-8, np.isfinite(Rn))
                norm_diff_real = (R1.real[mask] - Rn.real[mask]) / Rn.real[mask]
                max_diff = np.linalg.norm(norm_diff_real)
                RR1 = np.abs(R1) ** 2
                RRn = np.abs(Rn) ** 2
                norm_diff_RR = (RR1[mask] - RRn[mask]) / RRn[mask]
                max_diff_RR = np.linalg.norm(norm_diff_RR)
                print(f"{name} refl1d magnetic vs refl1d nsf: {max_diff:.2e}")
                print(f"{name} refl1d magnetic vs refl1d nsf RR: {max_diff_RR:.2e}")

    # assert np.linalg.norm((Rrefl1d[0] - Rpp) / Rpp) < 1e-13, "fail ++ %s" % name
    # assert np.linalg.norm((Rrefl1d[1] - Rpm) / Rpm) < 1e-13, "fail +- %s" % name
    # assert np.linalg.norm((Rrefl1d[2] - Rmp) / Rmp) < 1e-13, "fail -+ %s" % name
    # assert np.linalg.norm((Rrefl1d[3] - Rmm) / Rmm) < 1e-13, "fail -- %s" % name

    return kz, Rrefl1d


def simple():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 270, 0],
        [2000, 4.0, 2.0, 270, 0.0],
        [2000, 2.0, 2.0, 270, 0.0],
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


def validation_test_0():
    Aguide = 5
    # thickness	sld	mu	thetaM	sldm    roughness(ignore)
    # 0.000	0.000	0.000	90.00	0.000	0.000
    # 50.00	4.000	0.000	90.00	0.000	0.000
    # 825.0	9.060	0.000	90.00	0.000	0.000
    # 0.000	2.070	0.000	90.00	0.000	0.000
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 90.0, 0.0],
        [50, 4.0, 0.0, 90.0, 0.0],
        [825, 9.06, 0.0, 90.0, 0.0],
        [0, 2.07, 0.0, 90.0, 0.0],
    ]
    return "validation test 0", layers[::-1], Aguide


def validation_test_1():
    Aguide = 270
    # thickness	rho	sldi(ignore)	thetaM	rhoM    roughness
    # 0	4	0	90.00	0	0
    # 200	2	0	90.00	1	0
    # 200	4	0	180.0	1	0
    # 0	0	0	90.00	0	0
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 4.0, 0.0, 90.0, 0.0],
        [200, 2.0, 1.0, 90.0, 0.0],
        [200, 4.0, 1.0, 180.0, 0.0],
        [0, 0.0, 0.0, 90.0, 0.0],
    ]
    return "validation test 1", layers[::-1], Aguide


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


def spinflip_difference():
    Aguide = 0
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 1.0, 0.0],
        [200, 4.5, 2.0, 1.0, 0.0],
        [200, 4.5, 2.0, 1.0, 0.0],
        [0, 2.07, 0.0, 1.0, 0.0],
    ]
    return "SF difference", layers, Aguide


def Saerbeck_SmCo5(Aguide=270):
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 270, 0.0],
        [5.931, 8.01, 0.0, 270, 0.0],  # TaOx
        [24.4, 4.75, 0.0, 270, 0.0],  # Ta
        [144.3, 5.55, 3.594, 230, 0.0],  # Fe
        [240, 2.2365, 0.845, 210, 0.0],  # SmCo5
        [81.75, 3.1408, 0.0, 270, 0.0],  # Cr
        [0, 5.971, 0.0, 270, 0.0],  # MgO
    ]
    return "Saerbeck SmCo5", layers, Aguide


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


def _random_nsf_model(Nlayers=10, seed=None):
    if seed is None:
        seed = np.random.randint(1000000)
    np.random.seed(seed)
    dz = np.random.rand(Nlayers) * 2000.0 / Nlayers
    dz[0] = dz[-1] = 100.0
    rho = 8 * (np.random.rand(Nlayers) - 0.1)
    rhoM = 2 * np.random.rand(Nlayers)
    thetaM = 270.0 * np.ones(Nlayers)
    rhoM[0] = rhoM[-1] = 0.0
    phiM = np.zeros_like(dz)
    layers = np.vstack((dz, rho, rhoM, thetaM, phiM)).T.tolist()
    Aguide = 270.0
    # Aguide = 273.
    return "random_nsf(%d)" % seed, layers, Aguide


def simplest_pure_SF():
    Aguide = 0.00000001
    layers = [
        # depth rho rhoM thetaM phiM
        [0, 0.0, 0.0, 90.0, 0.0],
        [500, 8.0, 2.0, 90.0, 0.0],
        [0, 2.0, 0.0, 90.0, 0.0],
    ]
    return "simplest pure SF", layers, Aguide


def demo():
    """run demo"""
    with GeporeRunner() as runner:
        compare = functools.partial(run_comparison, runner)
        # compare(*simple(), zeeman_corrections=False)
        # compare(*simple(), zeeman_corrections=True, nsf_compare=True)
        # compare(*simplest_pure_SF(), H=0.5, zeeman_corrections=True)
        # compare(*simplest_pure_SF(), H=0.5, zeeman_corrections=False)
        # compare(*twist())
        # compare(*magsub())
        # compare(*magsurf())
        # compare(*zf_Yaohua_example(), H=0.4)  # 4000 gauss
        # compare(*zf_Yaohua_example(), H=0.0005)  # 5 Gauss
        # compare(*Kirby_example(), H=0.244)
        # compare(*zeeman_paper_example(), H=0.244)
        # compare(*zeeman_paper_example(), H=0)
        # compare(*spinflip_difference(), H=0.244)
        # compare(*Saerbeck_SmCo5(), H=0.300)
        # compare(*Saerbeck_SmCo5(Aguide=90), H=0.0)
        # compare(*NSF_example(), H=0.00005) # Earth's field, 0.5G
        # compare(*NSF_example(), H=1.0) # 1 tesla
        # compare(*Chuck_example(), H=0) # zeroish field, but magnetic front
        # compare(*_random_model(), H=0)
        # compare(*validation_test_0(), H=0.244)
        # compare(*validation_test_1(), H=0)
        # compare(*_random_model(), H=np.random.rand() * 2)
        # compare(*_random_model(seed=998543), H=2)
        seed = np.random.randint(1000000)
        compare(*_random_nsf_model(seed=seed), H=1.0, nsf_compare=True)
        compare(*_random_nsf_model(seed=seed), H=1.0, zeeman_corrections=False, nsf_compare=True)
        # compare(*_random_nsf_model(seed=998543), H=0.4, zeeman_corrections=True, nsf_compare=True)
        # plt.show()


def write_Chuck_result():
    kz, R = compare(*Chuck_example(), H=0)  # zeroish field, but magnetic front
    np.savetxt("Rmm.txt", np.vstack((2 * kz, np.abs(R[3]) ** 2)).T, delimiter="\t")
    np.savetxt("Rmp.txt", np.vstack((2 * kz, np.abs(R[2]) ** 2)).T, delimiter="\t")
    np.savetxt("Rpm.txt", np.vstack((2 * kz, np.abs(R[1]) ** 2)).T, delimiter="\t")
    np.savetxt("Rpp.txt", np.vstack((2 * kz, np.abs(R[0]) ** 2)).T, delimiter="\t")


if __name__ == "__main__":
    demo()
