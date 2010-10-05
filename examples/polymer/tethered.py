from refl1d.names import *
from copy import copy

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
D_initiator = SLD(name="D-initiator",rho=1.5)
H_toluene = SLD(name="H-toluene",rho=0.94)
H_initiator = SLD(name="H-initiator",rho=0)

D_brush = PolymerBrush(polymer=D_polystyrene, solvent=D_toluene,
                       base_vf=70, base=120, length=200, power=2,
                       sigma=10)

D = (silicon(0,5) | SiOx(100,5) | D_initiator(100,20) | D_brush(1000,0) 
     | D_toluene)

#### Undeuterated toluene solvent system
H_brush = copy(D_brush)  # Share tethered polymer parameters...
H_brush.solvent = H_toluene      # ... but use different solvent
H = silicon | SiOx | H_initiator | H_brush | H_toluene
for i,_ in enumerate(D):
    H[i].thickness = D[i].thickness
    H[i].interface = D[i].interface

# =============== fitted values ==================

if 0:
    D[0].interface.value = 9

    D[1].interface.value = 30
    D[1].thickness.value = 33
    
    D[2].interface.value = 7
    D_initiator.rho.value = 1.2
    D[2].thickness.value = 0
    
    D_brush.power.value = 1.93
    D_brush.base.value = 64
    D_brush.base_vf.value = 64
    D_polystyrene.rho.value = 6.42
    D_brush.length.value = 128
    D_brush.sigma.value = 5
    
    H_initiator.rho.value = 0.2


# ================= Fitting parameters ==================

for i in 0, 1, 2:
    D[i].interface.range(0,100)
D[1].thickness.range(0,200)
D[2].thickness.range(0,200)
D_polystyrene.rho.range(6.2,6.5)
SiOx.rho.range(2.07,4.16) # Si - SiO2
#SiOx.rho.pmp(10) # SiOx +/- 10%
D_toluene.rho.pmp(5)
D_initiator.rho.range(0,1.5)
D_brush.base_vf.range(50,80)
D_brush.base.range(0,100)
D_brush.length.range(0,500)
D_brush.power.range(1.5,2.5)
D_brush.sigma.range(0,20)

## Undeuterated system adds two extra parameters
H_toluene.rho.pmp(5)
H_initiator.rho.range(-0.5,0.5)


# ================= Data files ===========================
instrument = ncnrdata.NG7(Qlo=0.005, slits_at_Qlo=0.075)
D_probe = instrument.load('10ndt001.refl', back_reflectivity=True)
H_probe = instrument.load('10nht001.refl', back_reflectivity=True)


# ================== Model variations ====================
dream_opts = dict(chains=20,draws=300000,burn=1000000)
store = "T0" 
if len(sys.argv) > 1: store=sys.argv[1]
if store == "T1":
    dream_opts = dict(chains=20,draws=300000,burn=3000000)
    title = "decay tail"
elif store == "T2":
    dream_opts = dict(chains=20,draws=300000,burn=3000000)
    title = "decay tail; loose parameters (=T2)"
    D_toluene.rho.pmp(15)
    D_initiator.rho.range(-1,3)
    D_brush.base_vf.range(30,100)
    D_brush.base.range(0,200)
    D_brush.length.range(0,500)
    D_brush.power.range(1.,4.)
    D_brush.sigma.range(0,50)
else:
    raise RuntimeError("store %s not defined"%store)

# Join models and data
D_model = Experiment(sample=D, probe=D_probe)
H_model = Experiment(sample=H, probe=H_probe)
models = D_model, H_model

# Needed by dream fitter
modelnum = "all"
if len(sys.argv) > 2: modelnum=sys.argv[2]
if modelnum == "all":
    problem = MultiFitProblem(models=models)
    problem.name = "tethered"
elif modelnum == "M0":
    problem = FitProblem(D_model)
    problem.name = "Dtoluene"
elif modelnum == "M1":
    problem = FitProblem(H_model)
    problem.name = "Htoluene"
else:
    raise RuntimeError("model %s not defined"%modelnum)
problem.dream_opts = dream_opts
problem.title = title
problem.store = store
