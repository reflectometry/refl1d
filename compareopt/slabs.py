from random import getrandbits

from bumps.parameter import summarize
from numpy.random import uniform

from refl1d.names import *

num_layers = int(sys.argv[1])
init_file = sys.argv[2] if len(sys.argv) > 2 else "/tmp/problem"

out = open(init_file, "w")
seed = getrandbits(16)
print("seed", seed, file=out)
numpy.random.seed(int(seed))


# CONSTRAINTS="unknown"
CONSTRAINTS = "known"


def genlayer(name):
    material = SLD(name=name, rho=uniform(-3, 10))
    thickness = uniform(3, 200)
    interface = uniform(0, thickness / 3)
    layer = material(thickness, interface)
    if CONSTRAINTS == "unknown":
        layer.thickness.range(3, 200)
        layer.interface.range(0, 200.0 / 3)
        layer.material.rho.range(-3, 10)
    elif CONSTRAINTS == "known":
        layer.thickness.range(3, 2 * layer.thickness.value)
        layer.interface.range(0, min([2 * layer.thickness.value / 3.0, 200.0]))
        layer.material.rho.range(layer.material.rho.value - 1, layer.material.rho.value + 1)
    return layer


layers = [genlayer("L%d" % i) for i in range(num_layers)]
for i, L in layers[-1:1]:
    L.interface.value = min(
        [L.interface.value, L.thickness.value / 3, L[i].thickness.value / 3, L[i + 2].thickness.value / 3]
    )
layers[0].interface.value = min(
    [layers[0].interface.value, layers[0].thickness.value / 3, layers[1].thickness.value / 3]
)
layers[-1].interface.value = min(
    [layers[-1].interface.value, layers[-2].thickness.value / 3, layers[-1].thickness.value / 3]
)

sample = silicon(0, 5) | layers | air

T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = Experiment(probe=probe, sample=sample)
M.simulate_data(1.5)

problem = FitProblem(M)
print("chisq", problem.chisq(), file=out)
print("target", file=out)
print(summarize(problem.parameters), file=out)
problem.randomize()
M.update()
print("start", file=out)
print(summarize(problem.parameters), file=out)
out.close()
