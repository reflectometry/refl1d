from bumps.util import schema, List
from bumps.fitproblem import FitProblem as _FitProblem, FitProblemSchema as _FitProblemSchema
from .experiment import Experiment

# @schema(eq=False, init=False)
# class BaseFitProblem(_BaseFitProblem):
#     fitness: Experiment

@schema(classname="FitProblem", eq=False, init=False)
class FitProblemSchema(_FitProblemSchema):
    __doc__ =  _FitProblem.__doc__.replace("Fitness", "Experiment")
    models: List[Experiment]

class FitProblem(_FitProblem, FitProblemSchema):
    ...
