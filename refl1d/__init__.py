# This program is in the public domain
# Author: Paul Kienzle
"""
1-D Reflectometry Models

First define materials that you are going to use.

Common materials defined in :module:`materialdb`::

    *air*, *water*, *silicon*, *sapphire*, ...

Specific elements, molecules or mixtures can be added using the
classes in :module:`material`::

    *SLD*       unknown material with fittable SLD
    *Material*  known chemical formula and fittable density
    *Mixture*   known alloy or mixture with fittable fractions

Materials can be stacked as slabs, with a thickness for each layer and
roughness at the top of each layer.  Because this is such a common
operation, there is special syntax to do it, using '/' to specify
thickness and '%' to specify roughness.  For example, the following
is a 30 A gold layer on top of silicon, with a silicon:gold interface
of 5 A and a gold:air interface of 2 A::

    >> from refl1d import *
    >> sample = silicon%5 + gold/30%2 + air
    >> print sample
    Si + Au/30 + air

Individual layers and stacks can be used in multiple models, with all
parameters shared except those that are explicitly made separate.  The
syntax for doing so is similar to that for lists.  For example, the
following defines two samples, one with Si+Au/30+air and the other with
Si+Au/30+alkanethiol/10+air, with the silicon/gold layers shared::


    >> alkane_thiol = Material('C2H4OHS',bulk_density=0.8,name='thiol')
    >> sample1 = silicon%5 + gold/30%2 + air
    >> sample2 = sample1[:-1] + alkane_thiol/10%3 + air
    >> print sample2
    Si + Au/30 + thiol/10 + air

Stacks can be repeated using a simple multiply operation.  For example,
the following gives a cobalt/copper multilayer on silicon::

    >> Cu = Material('Cu')
    >> Co = Material('Co')
    >> sample = Si + (Co/30 + Cu/10)*20 + Co/30 + air
    >> print sample
    Si + (Co/30 + Cu/10)x20 + Co/30 + air

Multiple repeat sections can be included, and repeats can contain repeats.
Even freeform layers can be repeated.  By default the interface between
the repeats is the same as the interface between the repeats and the cap.
The cap interface can be set explicitly.  See :class:`model.Repeat` for
details.
"""

from periodictable import elements
from mystic import Parameter
from .version import __version__
from .experiment import Experiment
from .material import SLD, Material, Compound, Mixture
from .model import Slab, Stack
from .polymer import TetheredPolymer, VolumeProfile, layer_thickness
from .interface import Erf
from .probe import Probe, NeutronProbe, XrayProbe
from .fitter import preview, DEfit, SNOBfit
from . import ncnrdata, snsdata
fit = DEfit

# Pull in common materials for reflectometry experiments.
# This could lead to a lot of namespace pollution, and particularly to
# confusion if the user also does "from periodictable import *" since
# both of them create elements.
# Python doesn't allow "from .module import *"
from refl1d.materialdb import *


def plot_sample(sample, instrument=None, roughness_limit=0):
    """
    Quick plot of a reflectivity sample and the corresponding reflectivity.

    Use a data probe if the data is available.
    """
    if instrument == None:
        import numpy
        T = numpy.arange(0, 5, 0.05)
        probe = NeutronProbe(T=T, L=4.75)
    experiment = Experiment(sample=sample, probe=probe,
                            roughness_limit=roughness_limit)
    experiment.plot()
