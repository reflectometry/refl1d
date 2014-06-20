from refl1d.names import *

n = int(sys.argv[1]) if len(sys.argv)>1 else 2
materials = [SLD("L%d"%i,rho=1) for i in range(1,n+1)]
layers = [L(100,5) for L in materials]
sample = silicon(0,5) | layers | air

sample[0].interface.range(0,200)
for L in layers:
    L.material.rho.range(-10,10)
    L.thickness.range(0,1000)
    L.interface.range(0,200)

T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = Experiment(probe=probe, sample=sample)
problem = FitProblem(M)
