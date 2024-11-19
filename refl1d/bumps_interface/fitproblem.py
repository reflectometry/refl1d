from dataclasses import dataclass
from typing import List

from bumps.fitproblem import FitProblem as _FitProblem

from ..experiment import Experiment


# @dataclass(init=False)
# class BaseFitProblem(_BaseFitProblem):
#     fitness: Experiment


@dataclass(init=False)
class FitProblem(_FitProblem):
    """
    A dataclass which wraps bumps's FitProblem with a list of Experiment models.

    Mainly used to provide a more descriptive __doc__ string, and to allow
    IDEs to provide better type hints.
    """

    ### TODO: Replace with the following once the following issue is resolved:
    #   /refl1d/refl1d/bumps_interface/fitproblem.py:docstring of
    #   refl1d.bumps_interface.fitproblem.FitProblem:10: WARNING: undefined label: 'freevariables' [ref.ref]
    ###
    # __doc__ = _FitProblem.__doc__.replace("Fitness", "Experiment")
    models: List[Experiment]
