from ..deprecated.magnetic import FreeMagnetic, MagneticSlab, MagneticStack, MagneticTwist
from ..utils.support import sample_data
from .experiment import Experiment, MixedExperiment, plot_sample
from .probe.data_loaders import ncnrdata as NCNR
from .probe.data_loaders import snsdata as SNS
from .probe.data_loaders.load4 import load4
from .probe.data_loaders.stajconvert import load_mlayer, save_mlayer
from .probe.instrument import Monochromatic, Pulsed
from .probe.probe import (
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
from .sample.magnetism import FreeMagnetism, Magnetism, MagnetismStack, MagnetismTwist
from .sample.material import SLD, Compound, Material, Mixture
from .sample.mono import FreeInterface, FreeLayer
from .sample.polymer import EndTetheredPolymer, PolymerBrush, PolymerMushroom, VolumeProfile, layer_thickness

__all__ = [
    # from .deprecated.magnetic
    "FreeMagnetic",
    "MagneticSlab",
    "MagneticStack",
    "MagneticTwist",
    # from models
    "Experiment",
    "MixedExperiment",
    "plot_sample",
    "ChebyVF",
    "FreeformCheby",
    "cheby_approx",
    "cheby_points",
    "FunctionalMagnetism",
    "FunctionalProfile",
    "Monochromatic",
    "Pulsed",
    "NeutronProbe",
    "PolarizedNeutronProbe",
    "PolarizedQProbe",
    "Probe",
    "ProbeSet",
    "QProbe",
    "XrayProbe",
    "FreeMagnetism",
    "Magnetism",
    "MagnetismStack",
    "MagnetismTwist",
    "SLD",
    "Compound",
    "Material",
    "Mixture",
    "Slab",
    "Stack",
    "FreeInterface",
    "FreeLayer",
    "EndTetheredPolymer",
    "PolymerBrush",
    "PolymerMushroom",
    "VolumeProfile",
    "layer_thickness",
    # from readers
    "NCNR",
    "SNS",
    "load4",
    "load_mlayer",
    "save_mlayer",
    # from utils
    "sample_data",
]


# Deprecated names
def ModelFunction(*args, **kw):
    raise NotImplementedError("ModelFunction no longer supported --- use PDF instead")
