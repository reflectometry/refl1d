"""
Exported names

In model definition scripts, rather than importing symbols one by one, you
can simply perform:

    from refl1d.names import *

This is bad style for library and applications but convenient for
small scripts.
"""

import sys
import numpy as np

from periodictable import elements, formula
from bumps.parameter import Parameter, FreeVariables
from bumps import pmath
from bumps.pdfwrapper import PDF
from bumps.fitproblem import FitProblem
from bumps.fitproblem import MultiFitProblem  # deprecated

from .experiment import Experiment, plot_sample, MixedExperiment
from .flayer import FunctionalProfile, FunctionalMagnetism
from .material import SLD, Material, Compound, Mixture
from .model import Slab, Stack
from .polymer import (PolymerBrush, PolymerMushroom, EndTetheredPolymer,
                      VolumeProfile, layer_thickness)
from .mono import FreeLayer, FreeInterface
from .cheby import FreeformCheby, ChebyVF, cheby_approx, cheby_points
from .interface import Erf
from .probe import (Probe, ProbeSet, XrayProbe, NeutronProbe, QProbe,
                    PolarizedNeutronProbe, PolarizedQProbe, load4)
from .stajconvert import load_mlayer, save_mlayer
from . import ncnrdata as NCNR, snsdata as SNS
from .instrument import Monochromatic, Pulsed
from .magnetic import MagneticSlab, MagneticTwist, FreeMagnetic, MagneticStack
from .magnetism import Magnetism, MagnetismTwist, FreeMagnetism, MagnetismStack
from .support import sample_data

# Pull in common materials for reflectometry experiments.
# This could lead to a lot of namespace pollution, and particularly to
# confusion if the user also does "from periodictable import *" since
# both of them create elements.
# Python doesn't allow "from .module import *"
from .materialdb import *

# Deprecated names
def ModelFunction(*args, **kw):
    raise NotImplementedError("ModelFunction no longer supported --- use PDF instead")

PolarizedNeutronQProbe = PolarizedQProbe
numpy = np
