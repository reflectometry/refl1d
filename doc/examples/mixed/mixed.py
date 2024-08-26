# Channel measurement
# ===================
#
# In this example we will look at a nickel grating on a silicon substrate
# using specular reflectivity. When the spacing within the grating is
# sufficiently large, this can be modeled to first order as the incoherent sum
# of the reflectivity on the plateau and the reflectivity on the valley floor.
# By adjusting the weight of two reflectivities, we should be able to
# determine the ratio of plateau width to valley width.
#
# .. plot::
#
#    from sitedoc import plot_model
#    plot_model('mixed.py')

# Since silicon and air are defined, the only material we need to
# define is nickel.

from refl1d.names import *

nickel = Material("Ni")

# We need two separate models, one with 1000 |Ang| nickel and one without.

plateau = silicon(0, 5) | nickel(1000, 200) | air
valley = silicon(0, 5) | air

# We need only one probe for simulation.  The reflectivity measured at
# the detector will be a mixture of those neutrons which reflect off
# the plateau and those that reflect off the valley.

T = numpy.linspace(0, 2, 200)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

# We are going to start with a 1:1 ratio of plateau to valley and create
# a simulated data set.

M = MixedExperiment(samples=[plateau, valley], probe=probe, ratio=[1, 1])
M.simulate_data(5)

# We will assume the silicon interface is the same for the valley as the
# plateau, which depending on the how the sample is constructed, may or
# may not be realistic.

valley[0].interface = plateau[0].interface

# We will want to fit the thicknesses and interfaces as usual.

plateau[0].interface.range(0, 200)
plateau[1].interface.range(0, 200)
plateau[1].thickness.range(200, 1800)

# The ratio between the valley and the plateau can also be fit, either
# by fixing size of the plateau and fitting the size of the valley or
# fixing the size of the valley and fitting the size of the plateau.  We
# will hold the plateau fixed.

M.ratio[1].range(0, 5)

# Note that we could include a second order effect by including a
# hillside term with the same height as the plateau but using a
# 50:50 mixture of air and nickel.  In this case we would have three
# entries in the ratio.

# We wrap this as a fit problem as usual.

problem = FitProblem(M)

# This complete model script is defined in
# :download:`mixed.py <mixed.py>`:
#
# .. literalinclude:: mixed.py
#
# We can test how well the fitter can recover the original model
# by running refl1d with --random:
#
# .. parsed-literal::
#
#    $ refl1d mixed.py --random --store=T1
