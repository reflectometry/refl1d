"""
Exported names

In model definition scripts, rather than importing symbols one by one, you
can simply perform:

    from refl1d.names import *

This is bad style for library and applications but convenient for
small scripts.
"""

import numpy as np
from bumps import pmath
from bumps.fitproblem import FitProblem, MultiFitProblem
from bumps.parameter import FreeVariables, Parameter
from bumps.pdfwrapper import PDF
from periodictable import elements, formula

from .deprecated.magnetic import FreeMagnetic, MagneticSlab, MagneticStack, MagneticTwist
from .models.experiment import Experiment, MixedExperiment, plot_sample
from .models.sample.cheby import ChebyVF, FreeformCheby, cheby_approx, cheby_points
from .models.sample.flayer import FunctionalMagnetism, FunctionalProfile
from .models.instrument import Monochromatic, Pulsed
from .models.probe import (
    NeutronProbe,
    PolarizedNeutronProbe,
    PolarizedQProbe,
    Probe,
    ProbeSet,
    QProbe,
    XrayProbe,
)
from .models.sample.magnetism import FreeMagnetism, Magnetism, MagnetismStack, MagnetismTwist
from .models.sample.material import SLD, Compound, Material, Mixture
from .models.sample.layers import Slab, Stack
from .models.sample.mono import FreeInterface, FreeLayer
from .models.sample.polymer import EndTetheredPolymer, PolymerBrush, PolymerMushroom, VolumeProfile, layer_thickness
from .readers import ncnrdata as NCNR
from .readers import snsdata as SNS
from .readers.load4 import load4
from .readers.stajconvert import load_mlayer, save_mlayer
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


PolarizedNeutronQProbe = PolarizedQProbe
numpy = np
