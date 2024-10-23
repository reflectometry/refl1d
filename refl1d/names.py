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

from periodictable import elements, formula
from bumps.parameter import Parameter, FreeVariables
from bumps import pmath
from bumps.pdfwrapper import PDF
from bumps.fitproblem import FitProblem, MultiFitProblem
# from bumps.fitproblem import FitProblem  # deprecated

from .models.experiment import Experiment, plot_sample, MixedExperiment
from .models.sample.flayer import FunctionalProfile, FunctionalMagnetism
from .models.sample.material import SLD, Material, Compound, Mixture
from .models.sample.layers import Slab, Stack
from .models.sample.polymer import PolymerBrush, PolymerMushroom, EndTetheredPolymer, VolumeProfile, layer_thickness
from .models.sample.mono import FreeLayer, FreeInterface
from .models.sample.cheby import FreeformCheby, ChebyVF, cheby_approx, cheby_points
from .models.probe.probe import (
    Probe,
    ProbeSet,
    XrayProbe,
    NeutronProbe,
    QProbe,
    PolarizedNeutronProbe,
    PolarizedQProbe,
)
from .models.probe.data_loaders.stajconvert import load_mlayer, save_mlayer
from .models.probe.data_loaders import load4, ncnrdata as NCNR, snsdata as SNS
from .models.probe.instrument import Monochromatic, Pulsed
from .deprecated.magnetic import MagneticSlab, MagneticTwist, FreeMagnetic, MagneticStack
from .models.sample.magnetism import Magnetism, MagnetismTwist, FreeMagnetism, MagnetismStack
from .utils.support import sample_data

# Pull in common materials for reflectometry experiments.
# This could lead to a lot of namespace pollution, and particularly to
# confusion if the user also does "from periodictable import *" since
# both of them create elements.
# Python doesn't allow "from .module import *"
from .models.sample.materialdb import *


# Deprecated names
def ModelFunction(*args, **kw):
    raise NotImplementedError("ModelFunction no longer supported --- use PDF instead")


logging.warning("\trefl1d.names is deprecated.  Use `refl1d.models.*` instead.")

PolarizedNeutronQProbe = PolarizedQProbe
numpy = np
