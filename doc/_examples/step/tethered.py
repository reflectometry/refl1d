from refl1d.names import *
from copy import copy

# === Materials ===
SiOx = SLD(name="SiOx",rho=3.47)
D_polystyrene = SLD(name="D-PS",rho=6.2)
toluene = SLD(name="H-toluene",rho=0.94)

# === Sample ===
# Deuterated sample
brush = PolymerBrush(polymer=D_polystyrene, solvent=toluene,
                     base_vf=70, base=120, length=80, power=2,
                     sigma=10)

sample = (silicon(0,5) | SiOx(100,5) | brush(400,0) | toluene)

T = numpy.linspace(0, 5, 400)
dT,L,dL = 0.02,4.75, 0.0475
probe = NeutronProbe(T=T, dT=dT, L=L, dL=dL)
M0 = Experiment(sample=sample, probe=probe,
                dz=.1, dA=.1,
                name="dz=.1; dA=.1; smooth interfaces")
M1 = Experiment(sample=sample, probe=probe,
                dz=1, dA=1,
                name="dz=1; dA=1; smooth interfaces")
M2 = Experiment(sample=sample, probe=probe,
                dz=1, dA=10,
                name="dz=1; dA=10; smooth interfaces")
S0 = Experiment(sample=sample, probe=probe,
                dz=.1, dA=.1, step_interfaces=True,
                name="dz=.1; dA=.1; step interfaces")
S1 = Experiment(sample=sample, probe=probe,
                dz=1, dA=1, step_interfaces=True,
                name="dz=1; dA=1; step interfaces")
S2 = Experiment(sample=sample, probe=probe,
                dz=1, dA=10, step_interfaces=True,
                name="dz=1; dA=10; step interfaces")

models=[M0,M1,M2,S0,S1,S2]
#models=[S1]
problem = MultiFitProblem(models=models)
problem.name = "tethered"
