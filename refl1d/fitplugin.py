"""
Reflectivity plugin for fitting GUI.

Note that the fitting infrastructure is still heavily tied to the reflectivity
modeling program, and this represents only the first tiny steps to separating
the two.
"""

__all__ = ["data_view", "model_view", "new_model", "calc_errors", "show_errors"]

import numpy

from . import names as refl
from .errors import calc_errors, show_errors

# These are names used by the driver
def data_view():
    from .view.data_view import DataView
    return DataView

def model_view():
    from .view.model_view import ModelView
    return ModelView

def load_model(filename):
    if (filename.endswith('.so') or filename.endswith('.dll')
            or filename.endswith('.dyld')):
        from . import garefl
        problem = garefl.load(filename)
        return problem
    elif filename.endswith('.staj') or filename.endswith('.sta'):
        from .stajconvert import load_mlayer
        return refl.FitProblem(load_mlayer(filename))
        #fit_all(problem.fitness, pmp=20)
    else:
        return None

def new_model():
    stack = refl.silicon(0, 10) | refl.air
    instrument = refl.NCNR.NG1()
    probe = instrument.probe(T=numpy.linspace(0, 5, 200),
                             Tlo=0.2, slits_at_Tlo=2)
    M = refl.Experiment(sample=stack, probe=probe)
    problem = refl.FitProblem(M)
    return problem
