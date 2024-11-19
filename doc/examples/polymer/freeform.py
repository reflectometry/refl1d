# Freeform interface
# ==================

# Rather than using a specific model for the polymer brush we can use a
# freeform interface which varies the density between layers using a
# cubic spline interface.

from refl1d.names import *
from copy import copy

# Materials used

D_polystyrene = SLD(name="D-PS", rho=6.2)
SiOx = SLD(name="SiOx", rho=3.47)
D_toluene = SLD(name="D-toluene", rho=5.66)
D_initiator = SLD(name="D-initiator", rho=1.5)
H_toluene = SLD(name="H-toluene", rho=0.94)
H_initiator = SLD(name="H-initiator", rho=0)

# Define the freeform interface

n = 5
D_polymer_layer = FreeInterface(below=D_polystyrene, above=D_toluene, dz=[1] * n, dp=[1] * n)

# Stack materials into samples

# Note: only need D_toluene to compute Fresnel-normalized reflectivity --- should fix
# this later so that we can use a pure freeform layer on top.
D = silicon(0, 5) | SiOx(100, 5) | D_initiator(100, 20) | D_polymer_layer(1000, 0) | D_toluene

# Undeuterated toluene solvent system
H_polymer_layer = copy(D_polymer_layer)  # Share tethered polymer parameters...
H_polymer_layer.above = H_toluene  # ... but use different solvent
H = silicon | SiOx | H_initiator | H_polymer_layer | H_toluene
for i, _ in enumerate(D):
    H[i].thickness = D[i].thickness
    H[i].interface = D[i].interface


# Fitting parameters

for i in (0, 1, 2):
    D[i].interface.range(0, 100)
D[1].thickness.range(0, 200)
D[2].thickness.range(0, 200)
D_polystyrene.rho.range(6.2, 6.5)
SiOx.rho.range(2.07, 4.16)  # Si - SiO2
# SiOx.rho.pmp(10) # SiOx +/- 10%
D_toluene.rho.pmp(5)
D_initiator.rho.range(0, 1.5)
for p in D_polymer_layer.dz[1:]:
    p.range(0, 1)

## Undeuterated system adds two extra parameters
H_toluene.rho.pmp(5)
H_initiator.rho.range(-0.5, 0.5)

# Data files

instrument = NCNR.NG7(Qlo=0.005, slits_at_Qlo=0.075)
D_probe = instrument.load("10ndt001.refl", back_reflectivity=True)
H_probe = instrument.load("10nht001.refl", back_reflectivity=True)

# Join models and data

D_model = Experiment(sample=D, probe=D_probe)
H_model = Experiment(sample=H, probe=H_probe)
models = D_model, H_model

problem = FitProblem(models=models)
