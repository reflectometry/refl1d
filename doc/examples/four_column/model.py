# Four column data
# ================
#
# This example reuses the spin-value model for a completely unrelated
# measurement.  The goal is to demonstrate loading of four column data
# files (Q, R, dR, dQ) produced by the NCNR reductus fitting program.

# The following is copied directly from the spin value example

from refl1d.names import *

Si = SLD(name="Si", rho=2.0737, irho=2.376e-5)
Cu = SLD(name="Cu", rho=6.5535, irho=8.925e-4)
Ta = SLD(name="Ta", rho=3.8300, irho=3.175e-3)
TaOx = SLD(name="TaOx", rho=1.6325, irho=3.175e-3)
NiFe = SLD(name="NiFe", rho=9.1200, irho=1.032e-3)
CoFe = SLD(name="CoFe", rho=4.3565, irho=7.986e-3) # 60:40
IrMn = SLD(name="IrMn", rho=-0.21646, irho=4.245e-2)

sample = (Si(0, 2.13) | Ta(38.8, 2)
          | NiFe(25.0, 5, magnetism=Magnetism(rhoM=1.4638, thetaM=270,
                                              interface_below=2,
                                              interface_above=3))
          | CoFe(12.7, 5, magnetism=Magnetism(rhoM=3.7340, thetaM=270,
                                              interface_above=4))
          | Cu(28, 2)
          | CoFe(30.2, 5, MagnetismTwist(rhoM=[4.5102, 1.7860], thetaM=[270, 85],
                                         interface_below=9,
                                         interface_above=7))
          | IrMn(4.74, 1.7)
          | Cu(5.148, 2) | Ta(55.4895, 2) | TaOx(47.42, 3.5) | air
         )

sample[2].thickness.pmp(20)
sample[2].magnetism.rhoM.pmp(20)

sample[2].magnetism.interface_below.range(0, 10)
sample[2].magnetism.interface_above.range(0, 10)
sample[3].magnetism.interface_above.range(0, 10)
sample[5].magnetism.interface_below.range(0, 10)
sample[5].magnetism.interface_above.range(0, 10)

# Here's the new loader.  Much simplified since the reduction computes the
# appropriate $\Delta Q$ for the data points, and we don't need to specify
# the slit openings and distances for the data set.  The options to the
# :func:`refl1d.probe.load4` function allow you to override things during
# load, such as the sample broadening of the resolution.

probe = load4("refl.txt")
experiment = Experiment(probe=probe, sample=sample, dz=0.3, dA=None, interpolation=10)
problem = FitProblem(experiment)
