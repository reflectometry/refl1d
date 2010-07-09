# Attached please find two data sets for a tethered  approximately 10 nm thick 
# deuterated polystyrene chains in deuterated and hydrogenated toluene. 
# 10 nm thickness is for dry conditions and I am assuming these chains will 
# swell to 14-18 nm thickness once they are in toluene.
#    10ndt is for deuterated toluene case
#    10nht is for hydrogenated toluene case
# I also have to tell you that these chains are bound to the substrate by 
# using an initiator layer between substrate and brush chains. So in your 
# model you should have a silicon layer, silicon oxide layer, initiator layer 
# which is mostly hydrocarbon and scattering length density should be between 
# 0 and 1.5 depending on how much solvent is in the layer. Then you have the 
# swollen brush chains and at the end bulk solvent. When we do these swelling 
# measurements beam penetrate the system from the silicon side and the bottom 
# layer is deuterated or hydrogenated toluene.

from periodictable import formula
from refl1d import *
from refl1d.fitter import MultiFitProblem
from copy import copy


## =============== Models ======================

## Materials composition based approach.
#deutrated_density = formula("C6H5C2D3").mass/formula("C6H5C2H3").mass
#D_polystyrene = Material("C6H5C2D3", density=0.909*deuterated_density)
#SiOx = Material("SiO2", density=2.634)
#alkane = Material("C8H18",density=0.703)  # Octane formula and density
#deutrated_density = formula("C6H5CD3").mass/formula("C6H5CH3").mass
#H_toluene = Material("C6H5CH3", density=0.8669)
#D_toluene = Material("C6H5CD3", density=0.8669*deuterated_density)
#H_initiator = Compound.byvolume(alkane, H_toluene, 10)
#D_initiator = Compound.byvolume(alkane, D_toluene, H_initiator.fraction[0])



### Deuterated toluene solvent system
D_polystyrene = SLD(name="D-PS",rho=6.2)
SiOx = SLD(name="SiOx",rho=3.47)
D_toluene = SLD(name="D-toluene",rho=5.66)
D_initiator = SLD(name="D-toluene+initiator",rho=1.5)

D_polymer_layer = TetheredPolymer(polymer=D_polystyrene, solvent=D_toluene,
                                  phi=100, head=50, tail=200, Y=2)

# Stack materials into samples
D = silicon%2 + SiOx/10%2 + D_initiator/5%2 + D_polymer_layer/500%0 + D_toluene



### Undeuterated toluene solvent system
H_toluene = SLD(name="H-toluene",rho=0.94)
H_initiator = SLD(name="H-toluene+initiator",rho=0)
H_polymer_layer = copy(D_polymer_layer)  # Share tethered polymer parameters...
H_polymer_layer.solvent = H_toluene      # ... but use different solvent
H = silicon + SiOx + H_initiator + H_polymer_layer + H_toluene
for i,_ in enumerate(D):
    H[i].thickness = D[i].thickness
    H[i].interface = D[i].interface

# ================= Fitting parameters ==================

for i in 0, 1, 2:
    D[i].interface.range(0,100)
D[1].thickness.range(0,200)
D[2].thickness.range(0,200)
D_polystyrene.rho.range(6.2,6.5)
SiOx.rho.range(2.07,4.16) # Si - SiO2
D_toluene.rho.pmp(5)
D_initiator.rho.range(0,1.5)
D_polymer_layer.phi.range(0,100)
D_polymer_layer.head.range(0,200)
D_polymer_layer.tail.range(0,500)
D_polymer_layer.Y.range(0,4)

## Undeuterated system adds two extra parameters
H_toluene.rho.pmp(5)
H_initiator.rho.range(0,1.5)


# ================= Data files ===========================
instrument = ncnrdata.NG7(Qlo=0.005, slits_at_Qlo=0.075)
D_probe = instrument.load('10ndt001.refl', back_reflectivity=True)
H_probe = instrument.load('10nht001.refl', back_reflectivity=True)
D_model = Experiment(sample=D, probe=D_probe)
H_model = Experiment(sample=H, probe=H_probe)
models = D_model, H_model


# ================== Model variations ====================
dream_opts = dict(chains=20,draws=300000,burn=1000000)
store = "T1"
if store == "T1":
    title = "First try"
else:
    raise RuntimeError("store %s not defined"%store)

# Needed by dream fitter
problem = MultiFitProblem(models=models)
problem.dream_opts = dream_opts
#problem.name = variant
problem.title = title
problem.store = store
