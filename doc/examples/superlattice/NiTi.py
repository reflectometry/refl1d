# Hard material structures
# =========================
#
# Here is an example of a multilayer system in the literature:
#
#    Singh, S., Basu, S., Bhatt, P., Poswal, A.K.,
#    Phys. Rev. B, 79, 195435 (2009)
#
# In this paper, the authors are interested in the interdiffusion properties
# of Ni into Ti through x-ray and neutron reflectivity measurements. The
# question of alloying at metal-metal interfaces at elevated temperatures is
# critically important for device fabrication and reliability.
#
# The model is defined in :download:`NiTi.py <NiTi.py>`.
#
# .. plot::
#
#    from sitedoc import plot_model
#    plot_model('NiTi.py')
#

# First define the materials we will use

from refl1d.names import *

nickel = Material("Ni")
titanium = Material("Ti")

# Next we will compose nickel and titanium into a bilayer and use that
# bilayer to define a stack with 10 repeats.

# Superlattice description
bilayer = nickel(50, 5) | titanium(50, 5)
sample = silicon(0, 5) | bilayer * 10 | air

# We allow the thickness to vary by +/- 100%

# Fitting parameters
bilayer[0].thickness.pmp(100)
bilayer[1].thickness.pmp(100)

# The interfaces vary between 0 and 30 |Ang|. The interface between repeats is
# defined by the interface at the top of the repeating stack, which in this case
# is the Ti interface.  The interface between the superlattice and the next
# layer is an independent parameter, whose value defaults to the same initial
# value as the interface between the repeats.

bilayer[0].interface.range(0, 30)
bilayer[1].interface.range(0, 30)
sample[0].interface.range(0, 30)
sample[1].interface.range(0, 30)

# If we wanted to have the interface for Ti between repeats identical to
# the interface between Ti and air, we could have tied the parameters
# together, but we won't in this example::
#
#   # sample[1].interface = bilayer[1].interface
#
#
#
# If instead we wanted to keep the roughness independent, but start with
# a different initial value, we could simply set the interface parameter
# value.  In this case, we are setting it to 10 |Ang|\ ::
#
#   # sample[1].interface.value = 10

# We can also fit the number of repeats.  This is not realistic in this
# example (the sample grower surely knows the number of layers in a
# sample like this), so we do so only to demonstrate how it works.

sample[1].repeat.range(5, 15)

# Before we can view the reflectivity, we must define the Q range over
# which we want to simulate, and combine this probe with the sample.

T = numpy.linspace(0, 5, 100)
probe = XrayProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
M = Experiment(probe=probe, sample=sample)
M.simulate_data(5)
problem = FitProblem(M)
