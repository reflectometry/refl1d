from refl1d.names import *

# Materials
nickel = Material('Ni')

ridge = silicon(0,5) | nickel(1000,5) | air
valley = silicon(0,5) | air

T = numpy.linspace(0, 2, 200)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
M = MixedExperiment(samples=[ridge,valley], probe=probe, ratio=[1,1])
M.simulate_data(5)

problem = FitProblem(M)