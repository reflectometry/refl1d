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
from bumps import pmath
from bumps.fitproblem import FitProblem, MultiFitProblem
from bumps.parameter import FreeVariables, Parameter
from bumps.pdfwrapper import PDF
from periodictable import elements, formula

from . import ncnrdata as NCNR
from . import snsdata as SNS
from .cheby import ChebyVF, FreeformCheby, cheby_approx, cheby_points
from .experiment import Experiment, MixedExperiment, plot_sample
from .flayer import FunctionalMagnetism, FunctionalProfile
from .instrument import Monochromatic, Pulsed
from .magnetic import FreeMagnetic, MagneticSlab, MagneticStack, MagneticTwist
from .magnetism import FreeMagnetism, FreeMagnetismInterface, Magnetism, MagnetismStack, MagnetismTwist
from .material import SLD, Compound, Material, Mixture

# Pull in common materials for reflectometry experiments.
# This could lead to a lot of namespace pollution, and particularly to
# confusion if the user also does "from periodictable import *" since
# both of them create elements.
from .materialdb import *
from .model import Slab, Stack
from .mono import FreeInterface, FreeLayer
from .polymer import EndTetheredPolymer, PolymerBrush, PolymerMushroom, VolumeProfile, layer_thickness
from .probe import NeutronProbe, PolarizedNeutronProbe, PolarizedQProbe, Probe, ProbeSet, QProbe, XrayProbe, load4
from .stajconvert import load_mlayer, save_mlayer
from .support import sample_data


# Deprecated names
def ModelFunction(*args, **kw):
    raise NotImplementedError("ModelFunction no longer supported --- use PDF instead")


PolarizedNeutronQProbe = PolarizedQProbe
numpy = np
