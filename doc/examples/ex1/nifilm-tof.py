# Choosing an instrument
# ======================
#
# Let's modify the simulation to show how a 100 |Ang| nickel film might
# look if measured on the SNS Liquids reflectometer:
#
# .. plot::
#
#    from sitedoc import plot_model
#    plot_model('nifilm-tof.py')
#
# This model is defined in :download:`nifilm-tof.py <nifilm-tof.py>`
#
# The sample definition is the same:

from refl1d.names import *

nickel = Material("Ni")
sample = silicon(0, 5) | nickel(100, 5) | air

# Instead of using a generic probe, we are using an instrument definition
# to control the simulation.

instrument = SNS.Liquids()
M = instrument.simulate(
    sample,
    T=[0.3, 0.7, 1.5, 3],
    slits=[0.06, 0.14, 0.3, 0.6],
    uncertainty=5,
)

# The *instrument* line tells us to use the geometry of the SNS Liquids
# reflectometer, which includes information like the distance between the
# sample and the slits and the wavelength range.  We then simulate measurements
# of the sample for several different angles *T* (degrees), each with its
# own slit opening *slits* (mm).  The simulated measurement duration is
# such that the median relative error on the measurement $\Delta R/R$
# will match *uncertainty* (%).  Because the intensity $I(\lambda)$ varies
# so much for a time-of-flight measurement, the central points will be
# measured with much better precision, and the end points will be measured
# with lower precision.  See
# :meth:`Pulsed.simulate <refl1d.instrument.Pulsed.simulate>` for details
# on all simulation parameters.

# Finally, we bundle the simulated measurement as a fit problem which
# is used by the rest of the program.

problem = FitProblem(M)
