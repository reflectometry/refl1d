# Adding an SiOx layer
# ====================
#
# We will tweak the fitting model a little to add an SiOx layer between
# the silicon and the nickel.  There is no justification for doing so for
# this data (and indeeded, it sets the SiOx layer to almost pure Si), but
# it does demonstrate a way to form a mixture of two materials by volume
# :download:`nifilm-mixture.py <nifilm-mixture.py>`:

from refl1d.names import *

# Here is the mixture formula.  We are giving the mass density along with the
# chemical formula for each part, followed by the percentage for that part.
# We are giving it the name SiOx so we can reference it later.  Additional
# components woudl be added as material, fraction, material, fraction, ...
# The bulk material will sum the fractions to 100%.

SiOx = Mixture.byvolume("Si@2.329", "SiO2@26.5", 50, name="SiOx")

nickel = Material("Ni")
sample = silicon(0, 5) | SiOx(10, 2) | nickel(125, 10) | air

# The same fitting parameters as before...

sample["Ni"].thickness.pm(50)
sample["Si"].interface.range(0, 12)
sample["Ni"].interface.range(0, 20)

# ...with the addition of a volume fraction between 0 and 100% for
# the SiOx layer.  The thickness on this layer is not fitted in this
# example because the system is already overparameterized (the sample
# data was generated without an SiOx layer).

sample["SiOx"].interface.range(0, 12)
sample["SiOx"].thickness.range(0, 20)
sample["SiOx"].material.fraction[0].range(0, 100)

# The remainder of the problem setup is the same.

instrument = SNS.Liquids()
files = ["nifilm-tof-%d.dat" % d for d in (1, 2, 3, 4)]
probe = ProbeSet(instrument.load(f) for f in files)

M = Experiment(probe=probe, sample=sample)

problem = FitProblem(M)
