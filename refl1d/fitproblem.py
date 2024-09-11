from dataclasses import dataclass
from typing import List
from bumps.fitproblem import FitProblem as _FitProblem
from .experiment import Experiment

# @dataclass(init=False)
# class BaseFitProblem(_BaseFitProblem):
#     fitness: Experiment


@dataclass(init=False)
class FitProblem(_FitProblem):
    __doc__ = _FitProblem.__doc__.replace("Fitness", "Experiment")
    models: List[Experiment]
