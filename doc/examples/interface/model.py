from refl1d.names import *


interface = FreeInterface(below=silicon,above=air,dz=[.01,1,.01],dp=[.01,1,.01])
sample = silicon | interface(100) | air

T = numpy.linspace(0, 2, 200)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = Experiment(sample=sample, probe=probe, dz=0.2)

problem = FitProblem(M)
