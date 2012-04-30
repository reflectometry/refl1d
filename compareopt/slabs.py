from random import getrandbits
from numpy.random import uniform
from refl1d.names import *
from bumps.parameter import summarize


out = open("/tmp/problem","w")
seed = getrandbits(16)
print >>out,"seed",seed
numpy.random.seed(int(seed))


num_layers = int(sys.argv[1])

def genlayer(name):
    material = SLD(name=name, rho=uniform(-3,10))
    thickness = uniform(3,200)
    interface = uniform(0,thickness/3)
    layer = material(thickness,interface)
    layer.thickness.range(3,200)
    layer.interface.range(3,200./3)
    layer.material.rho.range(-3,10)
    return layer

layers = [genlayer("L%d"%i) for i in range(num_layers)]
for i,L in layers[-1:1]:
    L.interface.value = min([L.interface.value, L.thickness.value/3, L[i].thickness.value/3, L[i+2].thickness.value/3])
layers[0].interface.value = min([layers[0].interface.value, layers[0].thickness.value/3, layers[1].thickness.value/3])
layers[-1].interface.value = min([layers[-1].interface.value, layers[-2].thickness.value/3, layers[-1].thickness.value/3])

sample = silicon(0,5) | layers | air

T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = Experiment(probe=probe, sample=sample)
M.simulate_data(5)

problem = FitProblem(M)
print >>out,"target"
print >>out,summarize(problem.parameters)
problem.randomize()
M.update()
print >>out,"start"
print >>out,summarize(problem.parameters)
out.close()

