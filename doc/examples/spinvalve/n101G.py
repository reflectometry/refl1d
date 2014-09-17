# A model with magnetic structure

from refl1d.names import *

# We still need the nuclear structure, so define the materials.

Si = SLD(name="Si", rho=2.0737, irho=2.376e-5)
Cu = SLD(name="Cu", rho=6.5535, irho=8.925e-4)
Ta = SLD(name="Ta", rho=3.8300, irho=3.175e-3)
TaOx = SLD(name="TaOx", rho=1.6325, irho=3.175e-3)
NiFe = SLD(name="NiFe", rho=9.1200, irho=1.032e-3)
CoFe = SLD(name="CoFe", rho=4.3565, irho=7.986e-3) # 60:40
IrMn = SLD(name="IrMn", rho=-0.21646, irho=4.245e-2)

# The materials are stacked as usual, but the layers with magnetism have
# an additional magnetism property specified.  This example use
# :class:`refl1d.magnetism.Magnetism` to define a flat magnetic layer with
# the given magnetic scattering length density *rhoM* and angle *thetaM*.
#
# The magnetism is anchored to the corresponding nuclear layer, and by
# default will have the same thickness and interface.   The magnetic interface
# can be shifted relative to the nuclear interface using *dead_below* and
# *dead_above*.  These can be negative, allowing the magnetism to extend
# beyond the nuclear layer.  The magnetic interface can also be varied
# independently by using *interface_above* and *interface_below* as in the
# example below.   Note that interface_below is ignored # in consecutive
# layers, much like the nuclear layers, for which the interface attribute
# indicates the interface above.  Using *extent=2*, the single magnetism
# definition can extend over two consecutive layers.
#
# The :class:`refl1d.magnetism.MagnetismTwist` allows you to define a magnetic
# layer whose values of theta and rho change linearly throughout the layer.
# There are additional magnetism types defined in :mod:`reflid.magnetism`.
# Note that the current definition of interface only transitions smoothly
# into and out of layers with constant magnetism.  This behaviour may change
# in newer releases.

sample = (Si(0,2.13) | Ta(38.8,2)
          | NiFe(25.0,5, magnetism=Magnetism(rhoM=1.4638, thetaM=270,
                                             interface_below=2,
                                             interface_above=3))
          | CoFe(12.7,5, magnetism=Magnetism(rhoM=3.7340, thetaM=270,
                                             interface_above=4))
          | Cu(28,2)
          | CoFe(30.2, 5, MagnetismTwist(rhoM=[4.5102,1.7860], thetaM=[270,85],
                                         interface_below=9,
                                         interface_above=7))
          | IrMn(4.74,1.7)
          | Cu(5.148,2) | Ta(55.4895,2) | TaOx(47.42,3.5) | air
          )

# Define the fittable parameters as usual, including the magnetism attributes.

sample[2].thickness.pmp(20)
sample[2].magnetism.rhoM.pmp(20)

sample[2].magnetism.interface_below.range(0,10)
sample[2].magnetism.interface_above.range(0,10)
sample[3].magnetism.interface_above.range(0,10)
sample[5].magnetism.interface_below.range(0,10)
sample[5].magnetism.interface_above.range(0,10)

# Load the data

instrument = NCNR.NG1(slits_at_Tlo=0.1)
probe = instrument.load_magnetic("n101Gc1.reflA")

# We are going to compare the calculated reflectivity given two different
# step sizes on the profile.  Steps of *dz=0.3* are good enough for this
# example in that finer steps will not significantly change :math:`\chi^2`
# Steps of *dz=2* however are significantly different.  You can see the
# difference by looking at the spin asymmetry curves for the model rendered
# with *dz=2* and *dz=0.3* as we do below.  The reflectivity calculation time
# scales linearly with the step size, so you may want to use a large step
# size for your initial fits and a smaller step size later.  The *dA* parameter
# ought to give the best of both worlds, using a finer step size where the
# profile is changing quickly and coarser step size elsewhere, but it is
# currently broken and disabled below.

experiment = Experiment(probe=probe, sample=sample, dz=0.3, dA=None)
experiment2 = Experiment(probe=probe, sample=sample, dz=2, dA=None)
problem = FitProblem([experiment,experiment2])
