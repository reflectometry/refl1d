from refl1d.names import *
import periodictable
from copy import copy

numpy.random.seed(1)

# Start with an SiOx layer of pure silicon so there is no contrast match.
# Use this to simulate the layer.

# === Materials ===
Si_rho, Si_irho, Si_inc = periodictable.Si.neutron.sld(wavelength=4.75)
SiOx = SLD(name="SiOx",rho=Si_rho,irho=Si_irho)
D_toluene = SLD(name="D-toluene",rho=5.66)
D_initiator = SLD(name="D-initiator",rho=1.5)
D_polystyrene = SLD(name="D-PS",rho=6.2)
H_toluene = SLD(name="H-toluene",rho=0.94)
H_initiator = SLD(name="H-initiator",rho=0)

# === Sample ===
# Deuterated sample
D_brush = PolymerBrush(polymer=D_polystyrene, solvent=D_toluene,
                       base_vf=70, base=120, length=80, power=2,
                       sigma=10)

D = (silicon(0,5) | SiOx(50,5) | D_initiator(100,20) | D_brush(400,0)
     | D_toluene)


# === Fit parameters ===
for i in 0, 1, 2:
    D[i].interface.range(0,100)
D[1].thickness.range(0,200)
D[2].thickness.range(0,200)
D_polystyrene.rho.range(6.2,6.5)
SiOx.rho.range(2.07,4.16) # Si to SiO2
D_toluene.rho.pmp(5)
D_initiator.rho.range(0,1.5)
D_brush.base_vf.range(50,80)
D_brush.base.range(0,200)
D_brush.length.range(0,500)
D_brush.power.range(0,5)
D_brush.sigma.range(0,20)

# Define the probe and simulate data with 5% noise.

T = numpy.linspace(0, 5, 200)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
M = Experiment(sample=D, probe=probe)
M.simulate_data(noise=5)

# Wrap the result in a fit problem.

problem = FitProblem(M)
