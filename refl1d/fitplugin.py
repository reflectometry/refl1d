"""
Reflectivity plugin for fitting GUI.

Note that the fitting infrastructure is still heavily tied to the reflectivity
modeling program, and this represents only the first tiny steps to separating
the two.
"""

__all__ = [ "data_view", "model_view", "new_model",
            "calc_errors", "show_errors" ]

import refl1d.names as refl

# These are names used by the driver
from refl1d.errors import calc_errors, show_errors

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
        options = []
        return garefl.load(filename)
    elif filename.endswith('.staj'):
        from .stajconvert import load_mlayer, fit_all
        options = []
        return FitProblem(load_mlayer(filename))
        #fit_all(problem.fitness, pmp=20)
    else:
        return None

def new_model():
    stack = refl.silicon(0,10) | refl.air
    instrument = refl.NCNR.NG1()
    probe = instrument.probe(T=numpy.linspace(0,5,200),
                             Tlo=0.2, slits_at_Tlo=2)
    M = refl.Experiment(sample=stack, probe=probe)
    problem = refl.FitProblem(M)
    return problem
