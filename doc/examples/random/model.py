# Random model
# ============
#
# Generate a completely random film on Si to test fitting.
#
# For example, the following generates a random film with three layers::
#
#   refl1d model.py 3 --preview
#
# The model can also accept a noise level and a random number seed.
# Noise defaults to 3%.  If no seed is given, a random seed is generated
# and printed so that the model can be regenerated.
#
# To test the fitting engine, you will want to use --shake to set
# a random initial value before starting the fit:
#
#   refl1d model.py 3 --shake --fit=amoeba
#
# You will find that the amoeba fitter does not work well for
# random models.  Dream performs a bit better, able to recover
# models of 1-2 layers.
#
# The --simrandom method is not very good for reflectometry models,
# where we would rather have layer thicknesses distributed as
# exponential values (occasional thick layers, lots of thinner
# layers), and with roughness small compared to the layer thickness.
# The --simrandom will still work, overriding the parameters we generate
# with uniformly distributed values.
#
# There may be a more realistic choice for generated rho values than
# uniform in [-2, 10]; this may provide an unusual amount of contrast.
# Still, it is a good enough starting point, and does lead to some
# models with low contrast in neighbouring layers.

from refl1d.names import *

# Process command line arguments to the model

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
noise = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
seed = int(sys.argv[3]) if len(sys.argv) > 3 else np.random.randint(1, 9999)

# Set the seed for the random number generator.  Later we will print the
# seed, even if it was not set explicitly, so that interesting profiles
# can be regenerated.

np.random.seed(seed)

# Set up a model with the desired number of layers.  We will set the layer
# thickness and interfaces later.

materials = [SLD("L%d" % i, rho=1) for i in range(1, n + 1)]
layers = [L(100, 5) for L in materials]
sample = silicon(0, 5) | layers | air

# Set unlimited parameter ranges on those layers.

sample[0].interface.range(0, 200)
for L in layers:
    L.material.rho.range(-2, 10)
    L.thickness.range(0, 1000)
    L.interface.range(0, 200)

# Define the Q values at which to evaluate the model

T = numpy.linspace(0.1, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
M = Experiment(probe=probe, sample=sample)
problem = FitProblem(M)

# Set random values for rho.  This also sets thickness and interfaces, but
# these will be ignored.

problem.randomize()

# Generate layer thicknesses, with film thickness of about 400, but lots of
# variability in layer sizes.  Layers are limited to 950 so that the fit range
# can work.  Exponential distribution isn't suitable for single layer systems

for L in layers:
    L.thickness.value = min(np.random.exponential(400.0 / np.sqrt(n)), 950) if n > 1 else np.random.uniform(5, 950)

# Set interface limits based on neighbouring layer thickness, with substrate
# and surface having infinite thickness.  Choose an interface of at least 1 A

interfaces = [
    min(sample[i].thickness.value if i > 0 else np.inf, sample[i + 1].thickness.value if i < n else np.inf)
    for i in range(n + 1)
]
for L, w in zip(sample[: n + 1], interfaces):
    L.interface.value = 1 + np.random.exponential(w / 7)
    # Update the fit range if interface is excessively broad
    if L.interface.value > 200:
        L.interface.range(0, 2 * L.interface.value)

# Finally, generate some data with noise.

problem.simulate_data(noise=noise)
print("seed: %d" % seed)
print("target chisq: %s" % problem.chisq_str())
print(problem.summarize())
