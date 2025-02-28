"""
Reflectivity plugin for fitting GUI.

Note that the fitting infrastructure is still heavily tied to the reflectivity
modeling program, and this represents only the first tiny steps to separating
the two.
"""

__all__ = ["data_view", "model_view", "new_model", "calc_errors", "show_errors"]

from typing import cast
import numpy as np

from .migrations import migrate
from bumps.fitproblem import FitProblem
from ..experiment import Experiment
from ..sample.materialdb import air, silicon
from ..probe.data_loaders import ncnrdata as NCNR
from ..uncertainty import calc_errors, show_errors

# List of modules that contain dataclasses for the saved json file format


# These are names used by the driver
def data_view():
    from refl1d.wx_gui.data_view import DataView

    return DataView


def model_view():
    from refl1d.wx_gui.model_view import ModelView

    return ModelView


def load_model(filename):
    # TODO: bumps plugin api needs to allow options for loader
    options = None
    if filename.endswith(".staj") or filename.endswith(".sta"):
        from ..probe.data_loaders.stajconvert import load_mlayer

        return FitProblem[Experiment](load_mlayer(filename))
        # fit_all(problem.fitness, pmp=20)
    elif filename.endswith(".zip"):
        from bumps.fitproblem import load_problem

        # Note: bumps.vfs.vfs_init() must be called very early
        try:
            from bumps.vfs import ZipFS
        except ImportError:
            raise NotImplementedError("need newer bumps to load model from zip")
        with ZipFS(filename) as zf:
            for f in zf.filelist:
                if f.filename.endswith(".py"):
                    return cast(FitProblem[Experiment], load_problem(f.filename, options=options))
    elif filename.endswith(".json"):
        from bumps.serialize import load_file

        return cast(FitProblem[Experiment], load_file(filename))
    else:
        return None


def save_json(problem, basename):
    from bumps.serialize import save

    json_filename = basename + "-expt.json"
    save(json_filename, problem)


def new_model():
    stack = silicon(0, 10) | air
    instrument = NCNR.NG1()
    probe = instrument.probe(T=np.linspace(0, 5, 200), Tlo=0.2, slits_at_Tlo=2)
    M = Experiment(sample=stack, probe=probe)
    problem = FitProblem(M)
    return problem


def migrate_serialized(model_dict):
    _, migrated = migrate(model_dict)
    return migrated
