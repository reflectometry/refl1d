# Back reflectivity
# =================
#
# For samples measured with the incident beam through the substrate rather
# than reflecting off the surface, we don't need to modify our sample, we
# just need to tell the experiment that we are measuring back reflectivity.
#
# We set up the example as before.

from refl1d.names import *

nickel = Material("Ni")
sample = silicon(0, 25) | nickel(100, 5) | air
T = numpy.linspace(0, 5, 100)

# Because we are measuring back reflectivity, we create a probe which has
# back_reflectivity = True.

probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475, back_reflectivity=True)

# The remainder of the model definition is unchanged.

M = Experiment(probe=probe, sample=sample)
M.simulate_data(5)
problem = FitProblem(M)
