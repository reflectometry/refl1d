from typing import Union, Dict, List, TypedDict
import numpy as np
from refl1d.experiment import Experiment, ExperimentBase, MixedExperiment
import refl1d.probe
from refl1d.material import SLD
from bumps.webview.server.api import (
    register, get_chisq, state, to_json_compatible_dict, set_problem, publish
)
from bumps.errplot import calc_errors_from_state
# from refl1d.errors import show_errors
from .profile_uncertainty import show_errors

from .profile_plot import plot_multiple_sld_profiles, ModelSpec

# state.problem.serializer = "dataclass"

@register
async def get_plot_data(view: str = 'linear'):
    # TODO: implement view-dependent return instead of doing this in JS
    # (calculate x,y,dy.dx for given view, excluding log)
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    chisq = get_chisq(fitProblem)
    plotdata = []
    result = {"chisq": chisq, "plotdata": plotdata}
    for model in fitProblem.models:
        assert(isinstance(model, ExperimentBase))
        theory = model.reflectivity()
        probe = model.probe
        plotdata.append(get_probe_data(theory, probe, model._substrate, model._surface))

    return to_json_compatible_dict(result)

@register
async def get_profile_plots(model_specs: List[ModelSpec]):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    plot_items = []
    color_index = 0
    for model_index, model in enumerate(fitProblem.models):
        for sample_index, part in enumerate(getattr(model, 'parts', [model])):
            spec = dict(model_index=model_index, sample_index=sample_index)
            if spec in model_specs:
                plot_item = dict(model=part, spec=spec, color_index=color_index)
                plot_items.append(plot_item)
            color_index += 1

    fig = plot_multiple_sld_profiles(plot_items)
    output = to_json_compatible_dict(fig.to_dict())
    del fig
    return output



def get_single_probe_data(theory, probe, substrate=None, surface=None, polarization=""):
    fresnel_calculator = probe.fresnel(substrate, surface)
    Q, FQ = probe.apply_beam(probe.calc_Q, fresnel_calculator(probe.calc_Q))
    Q, R = theory
    output: Dict[str, Union[str, np.ndarray]]
    assert isinstance(FQ, np.ndarray)
    if len(Q) != len(probe.Q):
        # Saving interpolated data
        output = dict(Q = Q, theory = R, fresnel=np.interp(Q, probe.Q, FQ))
    elif getattr(probe, 'R', None) is not None:
        output = dict(Q = probe.Q, dQ = probe.dQ, R = probe.R, dR = probe.dR, theory = R, fresnel = FQ, background=probe.background.value, intensity=probe.intensity.value)
    else:
        output = dict(Q = probe.Q, dQ = probe.dQ, theory = R, fresnel = FQ)
    output["polarization"] = polarization
    output["label"] = probe.label()
    return output

def get_probe_data(theory, probe, substrate=None, surface=None):
    if isinstance(probe, refl1d.probe.PolarizedNeutronProbe):
        output = []
        for xsi, xsi_th, suffix in zip(probe.xs, theory, ('--', '-+', '+-', '++')):
            if xsi is not None:
                output.append(get_single_probe_data(xsi_th, xsi, substrate, surface, suffix))
        return output
    else:
        return [get_single_probe_data(theory, probe, substrate, surface)]

@register
async def get_model_names():
    problem = state.problem.fitProblem
    if problem is None:
        return None
    output: List[Dict] = []
    for model_index, model in enumerate(problem.models):
        if isinstance(model, Experiment):
            output.append(dict(name=model.name, part_name=None, model_index=model_index, part_index=0))
        elif isinstance(model, MixedExperiment):
            for part_index, part in enumerate(model.parts):
                output.append(dict(name=model.name, part_name=part.name, model_index=model_index, part_index=part_index))
    return output


@register
async def get_profile_uncertainty_plot(auto_align: bool=True, align: float=0., nshown: int=5000, npoints: int=5000, random: bool=True, residuals: bool=False):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    uncertainty_state = state.fitting.uncertainty_state
    align_arg = 'auto' if auto_align else align
    if uncertainty_state is not None:
        import time
        start_time = time.time()
        print('queueing new profile uncertainty plot...', start_time)
        errs = calc_errors_from_state(fitProblem, uncertainty_state, nshown=nshown, random=random, portion=1.0)
        print('errors calculated: ', time.time() - start_time)
        fig = show_errors(errs, npoints=npoints, align=align_arg, residuals=residuals)
        print("time to render but not serialize...", time.time() - start_time)
        output = to_json_compatible_dict(fig.to_dict())
        del fig
        end_time = time.time()
        print("time to draw profile uncertainty plot:", end_time - start_time)
        return output
    else:
        return None
