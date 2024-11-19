import numpy

from refl1d.names import *

nickel = Material("Ni")

plateau = silicon(0, 5) | MagneticSlab(nickel(1000, 200), rhoM=3) | air
valley = silicon(0, 5) | air

T = numpy.linspace(0, 2, 200)
up, down = [NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475, name="name") for name in ("up", "down")]
probe = PolarizedNeutronProbe(xs=[down, None, None, up])

M = MixedExperiment(samples=[plateau, valley], probe=probe, ratio=[1, 1])
M.simulate_data(5)

valley[0].interface = plateau[0].interface

plateau[0].interface.range(0, 200)
plateau[1].stack[0].interface.range(0, 200)
plateau[1].stack[0].thickness.range(200, 1800)

M.ratio[1].range(0, 5)

problem = FitProblem(M)
