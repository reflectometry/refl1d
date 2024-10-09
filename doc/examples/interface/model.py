from refl1d.models import *


sharp = FreeInterface(below=silicon, above=gold, dz=[0.01, 1, 0.01], dp=[0.01, 1, 0.01])
smooth = FreeInterface(below=gold, above=air, dz=[1], dp=[1])
sample = silicon | sharp(100) | gold(20) | smooth(100) | air

T = numpy.linspace(0, 2, 200)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = Experiment(sample=sample, probe=probe, dz=0.2)

problem = FitProblem(M)
