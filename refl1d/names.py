"""
Exported names

In model definition scripts, rather than importing symbols one by one, you
can simply perform:

    from refl1d.names import *

This is bad style for library and applications but convenient for
small scripts.
"""

import logging
import sys

import numpy as np
from bumps import pmath
from bumps.fitproblem import FitProblem, MultiFitProblem
from bumps.parameter import FreeVariables, Parameter
from bumps.pdfwrapper import PDF
from periodictable import elements, formula

from .deprecated.magnetic import FreeMagnetic, MagneticSlab, MagneticStack, MagneticTwist
from .experiment import Experiment, MixedExperiment, plot_sample
from .probe.data_loaders import ncnrdata as NCNR
from .probe.data_loaders import snsdata as SNS
from .probe.data_loaders.load4 import load4
from .probe.data_loaders.stajconvert import load_mlayer, save_mlayer
from .probe.instrument import Monochromatic, Pulsed
from .probe import (
    NeutronProbe,
    PolarizedNeutronProbe,
    PolarizedQProbe,
    Probe,
    ProbeSet,
    QProbe,
    XrayProbe,
)
from .sample.cheby import ChebyVF, FreeformCheby, cheby_approx, cheby_points
from .sample.flayer import FunctionalMagnetism, FunctionalProfile
from .sample.layers import Slab, Stack
from .sample.magnetism import FreeMagnetism, FreeMagnetismInterface, Magnetism, MagnetismStack, MagnetismTwist
from .sample.material import SLD, Compound, Material, Mixture

# Pull in common materials for reflectometry experiments.
# This could lead to a lot of namespace pollution, and particularly to
# confusion if the user also does "from periodictable import *" since
# both of them create elements.
from .sample.materialdb import *
from .sample.mono import FreeInterface, FreeLayer
from .sample.polymer import EndTetheredPolymer, PolymerBrush, PolymerMushroom, VolumeProfile, layer_thickness
from .utils.support import sample_data


# Deprecated names
def ModelFunction(*args, **kw):
    raise NotImplementedError("ModelFunction no longer supported --- use PDF instead")


PolarizedNeutronQProbe = PolarizedQProbe
numpy = np
