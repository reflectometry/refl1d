# .. _gepore_zeeman_sf:
#
# Validation of the Zeeman effect with gepore_zeeman.f
# ====================================================
#
# This example demonstrates the use of the gepore_zeeman.f program
# (a modification of the original gepore.f published in the PNR book chapter)
# to validate
# a model with strong spin-flip scattering and a high magnetic field
# (such as measuring a strongly anisotropic sample in a high field that
# is not parallel to the magnetization).

# The model is a simple slab with a single layer of magnetic material on
# a non-magnetic substrate.

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
# * magnetic layer: :math:`\rho_N = 8.0 \times 10^{-6}` and :math:`\rho_M = 2.0 \times 10^{-6}`
#   (the value of :math:`\theta_M` doesn't matter when H is out-of-plane)
# * substrate: non-magnetic with nuclear SLD :math:`\rho_N = 2.0 \times 10^{-6}`

Aguide = 0.00000001  # nearly zero to avoid division by zero in gepore.f
layers = [
    # depth rho rhoM thetaM phiM
    [0, 0.0, 0.0, 90.0, 0.0],
    [500, 8.0, 2.0, 90.0, 0.0],
    [0, 2.0, 0.0, 90.0, 0.0],
]
depth, rho, rhoM, thetaM, phiM = list(zip(*layers))

# applied field, in Tesla:

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

from refl1d.sample.reflectivity import magnetic_amplitude

r1 = magnetic_amplitude(Qz / 2, depth, rho, 0, rhoM, thetaM, 0, Aguide, H)
R1 = np.abs(r1[::-1]) ** 2

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
#     Aguide = 0.00000001
#     layers = [
#         [0, 0.0, 0.0, 90.0, 0.0],
#         [500, 8.0, 2.0, 90.0, 0.0],
#         [0, 2.0, 0.0, 90.0, 0.0],
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
#     Aguide = 0.00000001
#     layers = [
#         [0, 0.0, 0.0, 90.0, 0.0],
#         [500, 8.0, 2.0, 90.0, 0.0],
#         [0, 2.0, 0.0, 90.0, 0.0],
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
#         ax.plot(Qz, 2*(Rg[i] - R1[i])/np.abs(Rg[i] + R1[i]), label=f"rel. diff {label}")
#     ax.set_ylabel("Relative Reflectivity difference")
#     ax.set_xlabel("2*kz_in")
#     ax.set_title("Difference between gepore and refl1d, normalized to sum")
#     ax.legend()
