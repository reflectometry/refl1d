# Freeform structures
# ===================
#
# The following is a freeform superlattice floating in a solvent
# and anchored with a tether molecule.  The tether is anchored via
# a thiol group to a multilayer of Si/Cr/Au.  The sulphur in the
# thiol attaches well to gold, but not silicon.  Gold will stick
# to chrome which sticks to silicon.
#
# Here is the plot using a random tether, membrane and tail group:
#
# .. plot::
#
#    from sitedoc import plot_model
#    plot_model('freeform.py')

# The model is defined by :download:`freeform.py <freeform.py>`.

# The materials are straight forward:

from refl1d.names import *

chrome = Material("Cr")
gold = Material("Au")
solvent = Material("H2O", density=1)


# The sample description is more complicated.  When we define a freeform
# layer we need to anchor the ends of the freeform layer to a known
# material.  Usually, this is just the material that makes up the preceding
# and following layer.  In case we have freeform layers connected to each
# other, though, we need an anchor material that controls the SLD at the
# connection point.  For this purpose we introduce the dummy material
# wrap

wrap = SLD(name="wrap", rho=0)

# Each section of the freeform layer has a different number of control
# points.  The value should be large enough to give the profile enough
# flexibility to match the data, but not so large that it over fits the
# data.  Roughly the number of control points is the number of peaks and
# valleys allowed.  We want a relatively smooth tether and tail, so we
# keep *n1* and *n3* small, but make *n2* large enough to define an
# interesting repeat structure.

n1, n2, n3 = 3, 9, 3

# Free layers have a thickness, horizontal control points *z* varying
# in $[0,1]$, real and complex SLD $\rho$ and $\rho_i$, and the material
# above and below.

tether = FreeLayer(
    below=gold, above=wrap, thickness=10, z=numpy.linspace(0, 1, n1 + 2)[1:-1], rho=numpy.random.rand(n1), name="tether"
)
bilayer = FreeLayer(
    below=wrap,
    above=wrap,
    thickness=80,
    z=numpy.linspace(0, 1, n2 + 2)[1:-1],
    rho=5 * numpy.random.rand(n2) - 1,
    name="bilayer",
)
tail = FreeLayer(
    below=wrap,
    above=solvent,
    thickness=10,
    z=numpy.linspace(0, 1, n3 + 2)[1:-1],
    rho=numpy.random.rand(n3),
    name="tail",
)

# With the predefined free layers, we can quickly define a stack, with
# the bilayer repeat structure.  Note that we are setting the thickness
# for the free layers when we define the layers, so there is no need to
# set it when composing the layers into a sample.

sample = silicon(0, 5) | chrome(20, 2) | gold(50, 5) | tether | bilayer * 10 | tail | solvent

# Finally, simulate the resulting model.

T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475, back_reflectivity=True)
M = Experiment(probe=probe, sample=sample, dA=5)
M.simulate_data(5)
problem = FitProblem(M)
