import sys

from numpy import *

from periodictable import elements, formula
from .mystic.parameter import Parameter
from .version import __version__
from .experiment import Experiment, plot_sample
from .material import SLD, Material, Compound, Mixture
from .model import Slab, Stack
from .polymer import PolymerBrush, VolumeProfile, layer_thickness
from .mono import FreeLayer, FreeInterface
from .cheby import FreeformCheby, ChebyVF, cheby_approx, cheby_points
from .interface import Erf
from .probe import Probe, ProbeSet, NeutronProbe, XrayProbe
from .fitter import preview, fit, DEFit, SnobFit, mesh, FitProblem, MultiFitProblem
try:
    from .sampler import draw_samples
except Exception, exc:
    import traceback
    print traceback.print_exc()
    print "==== exception ignored"
from .stajconvert import load_mlayer, save_mlayer
from . import ncnrdata, snsdata

# Pull in common materials for reflectometry experiments.
# This could lead to a lot of namespace pollution, and particularly to
# confusion if the user also does "from periodictable import *" since
# both of them create elements.
# Python doesn't allow "from .module import *"
from refl1d.materialdb import *
