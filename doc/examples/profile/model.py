# Functional Layers
# =================
#
# Reflectometry layers can be arbitrary functions.  This is a rather
# arbitrary example, with a sinusoidal nuclear profile and an exponential
# magnetic profile.  A simulated dataset is generated from the model.

import numpy as np
from numpy import sin, pi, log, exp, hstack
from bumps.util import push_seed

from refl1d.names import *

# FunctionalProfile and FunctionalMagnetism are already available from
# refl1d.names, but a couple of aliases make them a little easier to access.

from refl1d.models.sample.flayer import FunctionalProfile as FP
from refl1d.models.sample.flayer import FunctionalMagnetism as FM

# Define the nuclear profile function.
#
# The first parameter to the function is *z*, which is the points within
# the layer at which to evaluate the function. The *z* steps are controlled
# by the *dz* parameter to the *Experiment* definition, which defaults
# to $\min(5, \tfrac{1}{10} \tfrac{2 \pi}{Q_\text{max}})$ in angstroms.
# The remaining parameters become fittable parameters in the model.
#
# The returned value is a complex number whose real part is *rho* and whose
# imaginary part is *irho*.  This example is for neutron reflectometry
# which for the most part does not have a strong absorption cross section.


def nuc(z, period, phase):
    """Nuclear profile"""
    return sin(2 * pi * (z / period + phase))


# Define the magnetic profile.  Like the nuclear profile, the first parameter
# is *z* and the remaining parameters become fittable parameters.  The returned
# value is *rhoM* or the pair *rhoM, thetaM*, with *thetaM* defaulting to 0
# if it is not returned.  Either *rhoM* or *thetaM* can be constant.


def mag(z, z1, z2, M1, M2, M3):
    r"""Magnetic profile

    Return the following function:

    .. math::

        f(z) = \left{ \begin{array}{ll}
            C & \mbox{if } z < z_1 \\
            re^{kz} & \mbox{if } z_1 \leq z \leq z_2 \\
            az+b & \mbox{if } z > z_2
            \end{array} \right.

    where $C = M_1$, $r,k$ are set such that $re^{kz_1} = M_1$ and
    $re^{kz_2} = M_2$, and $a,b$ are set such that $az_2 + b = M_2$
    and $az_{\rm end} + b = M_3$.
    """
    # Make sure z1 > z2, swapping if they are different.  Note that in the
    # posterior probability this will set P(z1, z2)=P(z2, z1) always.
    if z1 > z2:
        z1, z2 = z2, z1
    C = M1
    k = (log(M2) - log(M1)) / (z2 - z1)
    r = M1 / exp(k * z1)
    a = (M3 - M2) / (z[-1] - z2)
    b = M2 - a * z2

    part1 = z[z < z1] * 0 + C
    part2 = r * exp(k * z[(z >= z1) & (z <= z2)])
    part3 = a * z[z > z2] + b
    return hstack((part1, part2, part3))


# Use these functions to define the functional layer.

flayer = FP(
    100, 0, name="sin", profile=nuc, period=10, phase=0.2, magnetism=FM(profile=mag, M1=1, M2=4, M3=5, z1=10, z2=40)
)

# The functional layer is a normal layer which can be stacked into
# the model.  *flayer.start* and *flayer.end* are materials objects
# whose rho/irho values correspond to the complex *rho + j irho*
# value returned by the function at the start and end of the layer.
# Similarly, *magnetism.start* and *magnetism.end* return a magnetic
# layer defined by the start and end of the magnetic profile.

sample = silicon(0, 5) | flayer | flayer.end(35, 15, magnetism=flayer.magnetism.end) | air

# Need to be able to compute the thickness of the functional magnetic
# layer, which unfortunately requires the layer stack and an index.
# The index can be layer number, layer name, or if there are multiple
# layers with the same name, (layer name, k), where the magnetism
# is attached to the kth layer.

flayer.magnetism.set_anchor(sample, "sin")

# Set the fittable parameters.  Note that the parameters to the function
# after the first parameter *z* become fittable parameters.

sample["sin"].period.range(0, 100)
sample["sin"].phase.range(0, 1)
sample["sin"].thickness.range(0, 1000)
sample["sin"].magnetism.M1.range(0, 10)
sample["sin"].magnetism.M2.range(0, 10)
sample["sin"].magnetism.M3.range(0, 10)
sample["sin"].magnetism.z1.range(0, 100)
sample["sin"].magnetism.z2.range(0, 100)

# Define the model.  Since this is a simulation, we need to define the
# incident beam in terms of angles, wavelengths and dispersion.  This
# gets attached to the model forming an experiment.  Finally, we simulate
# data for the experiment with 5% dR/R.  We set the seed for the simulation
# so that the result is reproducible.  We could instead set the seed to
# None so that it pulls a random seed from entropy.

T = np.linspace(0, 5, 100)
xs = [NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475, name=name) for name in ("--", "-+", "+-", "++")]
probe = PolarizedNeutronProbe(xs)
M = Experiment(probe=probe, sample=sample, dz=0.1)
with push_seed(1):
    M.simulate_data(5)

problem = FitProblem(M)
