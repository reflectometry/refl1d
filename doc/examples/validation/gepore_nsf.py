# .. _gepore_nsf:
#
# Validation of non-Spin-Flip scattering with gepore.f
# =====================================================
#
# We can check a simple magnetic model :math:`(\vec M \parallel \vec H)`
# with no spin-flip scattering against:
#
# * gepore.f as written directly in the PNR book chapter
# * gepore_zeeman.f (a slightly modified version of gepore.f that includes the Zeeman effect)
# * refl1d using the magnetic calculation kernel
# * refl1d using the unpolarized calculation kernel twice, with
#    * :math:`\rho_{++} = \rho_N + \rho_M`
#    * :math:`\rho_{--} = \rho_N - \rho_M`
#
# We are using the Refl1D geometry convention where
# :math:`\text{Aguide} = 270^\circ, \theta_M = 270^\circ` corresponds to an in-plane magnetic field
# with :math:`\vec M \parallel \vec H`.
# The sample will be a magnetic layer on a non-magnetic substrate, with a non-magnetic cap layer

import numpy as np
from refl1d.validation.gepore_runner import GeporeRunner

# start a GeporeRunner instance

runner = GeporeRunner()

QS = 0.001  # start value of Q
DQ = 0.0004  # step size in Q
NQ = 80  # number of Q points
Qz = np.arange(NQ) * DQ + QS

# Define the sample as a list of layers:
#
# * vacuum (all SLD zero)
# * magnetic layer: :math:`\rho_N = 8.0 \times 10^{-6}, \rho_M = 2.0 \times 10^{-6}, \theta_M = 270^\circ`
# * non-magnetic cap layer: :math:`\rho_N = 2.0 \times 10^{-6}`
# * substrate: non-magnetic with nuclear SLD :math:`\rho_N = 2.0 \times 10^{-6}`

Aguide = 270.0  # guide field in sample plane
layers = [
    # depth rho rhoM thetaM phiM
    [0, 0.0, 0.0, 270, 0],
    [1000, 8.0, 2.0, 270, 0.0],
    [500, 5.0, 0.0, 270, 0.0],
    [0, 2.0, 0.0, 270, 0.0],
]
depth, rho, rhoM, thetaM, phiM = list(zip(*layers))

# applied field, in Tesla, shouldn't matter for this calculation

H = 0.5

# Run the simulation with gepore, and retrieve the reflectivity amplitudes
# :math:`(rg^{++}, rg^{+-}, rg^{-+}, rg^{--})`
# note that we use a value of :math:`\text{EPS} = -\text{Aguide}`

EPS = -Aguide
rg = runner.run(layers, QS, DQ, NQ, EPS, H, zeeman_corrections=True)
Rg = [np.abs(r) ** 2 for r in rg]

# calculate the reflectivity using refl1d:
#
# magnetic_amplitude returns cross-sections in order :math:`(r^{--}, r^{-+}, r^{+-}, r^{++})`
# so we need to reverse them here to compare to gepore outputs

from refl1d.sample.reflectivity import magnetic_amplitude, reflectivity_amplitude

r1 = magnetic_amplitude(Qz / 2, depth, rho, 0, rhoM, thetaM, 0, Aguide, H)
R1 = np.abs(r1[::-1]) ** 2

# Calculating using unpolarized reflectivity, with rho + rhoM and rho - rhoM

rho_plus = rho + rhoM
rho_minus = rho - rhoM
sf = np.zeros_like(R1)
rnsf = (
    reflectivity_amplitude(Qz / 2, depth, rho_plus, 0, 0, None),
    sf,
    sf,
    reflectivity_amplitude(Qz / 2, depth, rho_minus, 0, 0, None),
)
Rnsf = [np.abs(r) ** 2 for r in rnsf]

# .. plot::
#
#     import numpy as np
#     from refl1d.validation.gepore_runner import GeporeRunner
#     from matplotlib import pyplot as plt
#     runner = GeporeRunner()
#     QS = 0.001 # start value of Q
#     DQ = 0.0004 # step size in Q
#     NQ = 80 # number of Q points
#     Qz = np.arange(NQ) * DQ + QS
#     Aguide = 270.0 # guide field in sample plane
#     layers = [
#         # depth rho rhoM thetaM phiM
#         [0, 0.0, 0.0, 270, 0],
#         [1000, 8.0, 2.0, 270, 0.0],
#         [500, 5.0, 0.0, 270, 0.0],
#         [0, 2.0, 0.0, 270, 0.0],
#     ]
#     depth, rho, rhoM, thetaM, phiM = list(zip(*layers))
#     H = 0.5
#     EPS = -Aguide
#     rg = runner.run(layers, QS, DQ, NQ, EPS, H, zeeman_corrections=True)
#     Rg = [np.abs(r)**2 for r in rg]
#     from refl1d.sample.reflectivity import magnetic_amplitude
#     r1 = magnetic_amplitude(Qz/2, depth, rho, 0, rhoM, thetaM, 0, Aguide, H)
#     R1 = np.abs(r1[::-1])**2
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     xs_labels = ["++", "+-", "-+", "--"]
#     for i, label in enumerate(xs_labels):
#         ax.plot(Qz, Rg[i], label=f"gepore {label}")
#     ax.set_prop_cycle(None)
#     for i, label in enumerate(xs_labels):
#         ax.plot(Qz, R1[i], 'o', label=f"refl1d {label}", fillstyle='none')
#     ax.set_ylabel("Reflectivity")
#     ax.set_xlabel("2*kz_in")
#     ax.legend()

# The differences between the two reflectivity outputs are small, and are
# likely due to differences in the numerical implementation of the
# reflectivity calculation.  Here is a plot of the differences:

# .. plot::
#
#     import numpy as np
#     from refl1d.validation.gepore_runner import GeporeRunner
#     from matplotlib import pyplot as plt
#     runner = GeporeRunner()
#     QS = 0.001 # start value of Q
#     DQ = 0.0004 # step size in Q
#     NQ = 80 # number of Q points
#     Qz = np.arange(NQ) * DQ + QS
#     Aguide = 270.0 # guide field in sample plane
#     layers = [
#         # depth rho rhoM thetaM phiM
#         [0, 0.0, 0.0, 270, 0],
#         [1000, 8.0, 2.0, 270, 0.0],
#         [500, 5.0, 0.0, 270, 0.0],
#         [0, 2.0, 0.0, 270, 0.0],
#     ]
#     depth, rho, rhoM, thetaM, phiM = list(zip(*layers))
#     H = 0.5
#     EPS = -Aguide
#     rg = runner.run(layers, QS, DQ, NQ, EPS, H, zeeman_corrections=True)
#     Rg = [np.abs(r)**2 for r in rg]
#     from refl1d.sample.reflectivity import magnetic_amplitude
#     r1 = magnetic_amplitude(Qz/2, depth, rho, 0, rhoM, thetaM, 0, Aguide, H)
#     R1 = np.abs(r1[::-1])**2
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     xs_labels = ["++", "+-", "-+", "--"]
#     for i, label in enumerate(xs_labels):
#         if not i == 1 and not i == 2: # skip +- and -+
#             ax.plot(Qz, 2*(Rg[i] - R1[i])/np.abs(Rg[i] + R1[i]), label=f"rel. diff {label}")
#     ax.set_ylabel("Relative Reflectivity difference")
#     ax.set_xlabel("2*kz_in")
#     ax.set_title("Difference between gepore and refl1d, normalized to sum")
#     ax.legend()

# .. plot::
#
#     import numpy as np
#     from refl1d.validation.gepore_runner import GeporeRunner
#     from matplotlib import pyplot as plt
#     runner = GeporeRunner()
#     QS = 0.001 # start value of Q
#     DQ = 0.0004 # step size in Q
#     NQ = 80 # number of Q points
#     Qz = np.arange(NQ) * DQ + QS
#     Aguide = 270.0 # guide field in sample plane
#     layers = [
#         # depth rho rhoM thetaM phiM
#         [0, 0.0, 0.0, 270, 0],
#         [1000, 8.0, 2.0, 270, 0.0],
#         [500, 5.0, 0.0, 270, 0.0],
#         [0, 2.0, 0.0, 270, 0.0],
#     ]
#     depth, rho, rhoM, thetaM, phiM = list(zip(*layers))
#     H = 0.5
#     EPS = -Aguide
#     rgz = runner.run(layers, QS, DQ, NQ, EPS, H, zeeman_corrections=True)
#     Rgz = [np.abs(r)**2 for r in rgz]
#     rg = runner.run(layers, QS, DQ, NQ, EPS, H, zeeman_corrections=False)
#     Rg = [np.abs(r)**2 for r in rg]
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     xs_labels = ["++", "+-", "-+", "--"]
#     for i, label in enumerate(xs_labels):
#         if not i == 1 and not i == 2: # skip +- and -+
#             ax.plot(Qz, 2*(Rgz[i] - Rg[i])/np.abs(Rgz[i] + Rg[i]), label=f"rel. diff {label}")
#     ax.set_ylabel("Relative Reflectivity difference")
#     ax.set_xlabel("2*kz_in")
#     ax.set_title("Difference between gepore and gepore_zeeman, normalized to sum")
#     ax.legend()

# .. plot::
#
#     import numpy as np
#     from matplotlib import pyplot as plt
#     QS = 0.001 # start value of Q
#     DQ = 0.0004 # step size in Q
#     NQ = 80 # number of Q points
#     Qz = np.arange(NQ) * DQ + QS
#     Aguide = 270.0 # guide field in sample plane
#     layers = [
#         # depth rho rhoM thetaM phiM
#         [0, 0.0, 0.0, 270, 0],
#         [1000, 8.0, 2.0, 270, 0.0],
#         [500, 5.0, 0.0, 270, 0.0],
#         [0, 2.0, 0.0, 270, 0.0],
#     ]
#     depth, rho, rhoM, thetaM, phiM = list(zip(*layers))
#     H = 0.5
#     from refl1d.sample.reflectivity import magnetic_amplitude, reflectivity_amplitude
#     r1 = magnetic_amplitude(Qz/2, depth, rho, 0, rhoM, thetaM, 0, Aguide, H)
#     R1 = np.abs(r1[::-1])**2
#     rho_plus = np.array(rho) + np.array(rhoM)
#     rho_minus = np.array(rho) - np.array(rhoM)
#     sf = np.zeros_like(R1[0])
#     rnsf = (
#         reflectivity_amplitude(Qz/2, depth, rho_plus, 0, 0, None),
#         sf,
#         sf,
#        reflectivity_amplitude(Qz/2, depth, rho_minus, 0, 0, None),
#     )
#     Rnsf = [np.abs(r) ** 2 for r in rnsf]
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     xs_labels = ["++", "+-", "-+", "--"]
#     for i, label in enumerate(xs_labels):
#         if not i == 1 and not i == 2: # skip +- and -+
#             ax.plot(Qz, 2*(Rnsf[i] - R1[i])/np.abs(Rnsf[i] + R1[i]), label=f"rel. diff {label}")
#     ax.set_ylabel("Relative Reflectivity difference")
#     ax.set_xlabel("2*kz_in")
#     ax.set_title("Difference of refl1d (mag. kernel) and\n refl1d (unpol. kernel twice), normalized to sum")
#     ax.legend()
