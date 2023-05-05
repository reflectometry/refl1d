import asyncio
from aiohttp import web
from dataclasses import dataclass
import json
from pathlib import Path
import socket
from typing import Union, Dict, List, Optional

import numpy as np
from bumps.webview.server import webserver
from bumps.webview.server.webserver import (
    app, sio, rest_get, state, get_chisq, to_json_compatible_dict, get_commandline_options,
    set_problem
)
from refl1d.experiment import Experiment, ExperimentBase, MixedExperiment
import refl1d.probe

# Register the refl1d model loader
import refl1d.fitplugin
import bumps.cli
bumps.cli.install_plugin(refl1d.fitplugin)

from .profile_plot import plot_sld_profile_plotly

client_path = Path(__file__).parent.parent / 'client'
index_path = client_path / 'dist'
static_assets_path = index_path / 'assets'

@dataclass
class Options(webserver.Options):
    serializer: webserver.SERIALIZERS = "dataclass"
    headless: bool = True
    app_name: str = "Refl1D"


async def index(request):
    """Serve the client-side application."""
    # check if the locally-build site has the correct version:
    with open(client_path / 'package.json', 'r') as package_json:
        client_version = json.load(package_json)['version'].strip()

    try:
        local_version = open(index_path / 'VERSION', 'rt').read().strip()
    except FileNotFoundError:
        local_version = None

    print(index_path, local_version, client_version, local_version == client_version)
    if client_version == local_version:
        return web.FileResponse(index_path / 'index.html')
    else:
        CDN = f"https://cdn.jsdelivr.net/npm/refl1d-webview-client@{client_version}/dist"
        with open(client_path / 'index_template.txt', 'r') as index_template:
            index_html = index_template.read().format(cdn=CDN)
        return web.Response(body=index_html, content_type="text/html")
    

@sio.event
@rest_get
async def get_plot_data(sid: str="", view: str = 'linear'):
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
        # fresnel_calculator = probe.fresnel(model._substrate, model._surface)
        # Q, FQ = probe.apply_beam(probe.calc_Q, fresnel_calculator(probe.calc_Q))
        # Q, R = theory
        # assert isinstance(FQ, np.ndarray)
        # if len(Q) != len(probe.Q):
        #     # Saving interpolated data
        #     output = dict(Q = Q, R = R, fresnel=np.interp(Q, probe.Q, FQ))
        # elif getattr(probe, 'R', None) is not None:
        #     output = dict(Q = probe.Q, dQ = probe.dQ, R = probe.R, dR = probe.dR, fresnel = FQ)
        # else:
        #     output = dict(Q = probe.Q, dQ = probe.dQ, R = R, fresnel = FQ)
        # result.append(output)
    return to_json_compatible_dict(result)

@sio.event
@rest_get
async def get_profile_plot(sid: str="", model_index: int=0, sample_index: int=0):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    models = list(fitProblem.models)
    if model_index > len(models):
        return None
    model = models[model_index]
    assert (isinstance(model, Union[Experiment, MixedExperiment]))
    if isinstance(model, MixedExperiment):
        model = model.parts[sample_index]
    fig = plot_sld_profile_plotly(model)
    output = to_json_compatible_dict(fig.to_dict())
    del fig
    return output



def get_single_probe_data(theory, probe, substrate=None, surface=None, label=''):
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
    output['label'] = f"{probe.label()} {label}"
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

@sio.event
async def get_model_names(sid: str=""):
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

def main(options: Optional[Options] = None, sock: Optional[socket.socket] = None):
    options = get_commandline_options(arg_defaults={"serializer": "dataclass", "headless": False}) if options is None else options
    options.app_name = "Refl1D"
    try:
        asyncio.run(start_app(options, sock))
    except KeyboardInterrupt:
        print("stopped by KeyboardInterrupt.")

async def start_app(options: Options = Options(), sock: Optional[socket.socket] = None):
    runsock = webserver.setup_app(options=options, static_assets_path=static_assets_path, index=index, sock=sock)
    await web._run_app(app, sock=runsock)

if __name__ == '__main__':
    main()