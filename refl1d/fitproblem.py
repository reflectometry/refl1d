from bumps.util import schema, List
from bumps.fitproblem import FitProblem as _FitProblem
from .experiment import Experiment

# @schema(eq=False, init=False)
# class BaseFitProblem(_BaseFitProblem):
#     fitness: Experiment

@schema(eq=False, init=False)
class FitProblem(_FitProblem):
    models: List[Experiment]