"""
Reflectivity plugin for fitting GUI.

Note that the fitting infrastructure is still heavily tied to the reflectivity
modeling program, and this represents only the first tiny steps to separating
the two.
"""

__all__ = ["data_view", "model_view", "new_model", "calc_errors", "show_errors"]

import numpy as np

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
    # TODO: bumps plugin api needs to allow options for loader
    options = None
    if (filename.endswith('.so') or filename.endswith('.dll')
            or filename.endswith('.dyld')):
        from . import garefl
        problem = garefl.load(filename)
        return problem
    elif filename.endswith('.staj') or filename.endswith('.sta'):
        from .stajconvert import load_mlayer
        return refl.FitProblem(load_mlayer(filename))
        #fit_all(problem.fitness, pmp=20)
    elif filename.endswith('.zip'):
        from bumps.fitproblem import load_problem
        # Note: bumps.vfs.vfs_init() must be called very early
        try:
            from bumps.vfs import ZipFS
        except ImportError:
            raise NotImplementedError("need newer bumps to load model from zip")
        with ZipFS(filename) as zf:
            for f in zf.filelist:
                if f.filename.endswith('.py'):
                    return load_problem(f.filename, options=options)
    else:
        return None

def new_model():
    stack = refl.silicon(0, 10) | refl.air
    instrument = refl.NCNR.NG1()
    probe = instrument.probe(T=np.linspace(0, 5, 200),
                             Tlo=0.2, slits_at_Tlo=2)
    M = refl.Experiment(sample=stack, probe=probe)
    problem = refl.FitProblem(M)
    return problem
