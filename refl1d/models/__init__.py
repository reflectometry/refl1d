from bumps.parameter import Parameter

from ..deprecated.magnetic import FreeMagnetic, MagneticSlab, MagneticStack, MagneticTwist
from ..utils.support import sample_data
from .bumps_interface.fitproblem import FitProblem
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
from .sample.layers import Repeat, Slab, Stack
from .sample.magnetism import FreeMagnetism, Magnetism, MagnetismStack, MagnetismTwist
from .sample.material import SLD, Compound, Material, Mixture, Vacuum
from .sample.materialdb import (
    D2O,
    DHO,
    H2O,
    Al2O3,
    Au,
    Ni8Fe2,
    Si,
    air,
    gold,
    heavywater,
    lightheavywater,
    permalloy,
    sapphire,
    silicon,
    water,
)
from .sample.mono import FreeInterface, FreeLayer
from .sample.polymer import EndTetheredPolymer, PolymerBrush, PolymerMushroom, VolumeProfile, layer_thickness


# Deprecated names
def ModelFunction(*args, **kw):
    raise NotImplementedError("ModelFunction no longer supported --- use PDF instead")
