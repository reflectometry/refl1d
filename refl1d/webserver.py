# from .main import setup_bumps

from typing import Any, Dict, List
from datetime import datetime
from aiohttp import web
import numpy as np
import asyncio
import socketio
from pathlib import Path, PurePath
import json
from copy import deepcopy

from bumps.fitters import DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit
from bumps.serialize import to_dict
import refl1d.fitproblem, refl1d.probe

FITTERS = (DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit)
FITTER_DEFAULTS = {}
for fitter in FITTERS:
    FITTER_DEFAULTS[fitter.id] = {
        "name": fitter.name,
        "settings": dict(fitter.settings)
    }

# set up mutable state of fitters:
# fitter_settings = deepcopy(FITTER_DEFAULTS)
# active_fitter = "amoeba"

sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
static_dir_path = Path(__file__).parent / 'webview'
sio.attach(app)

# class TopicManager:
#     topics: Dict[str, Any] = {}

#     def subscribe(self, sid: str, topic: str):
#         sio.enter_room(sid, topic)

#     def unsubscribe(self, sid: str, topic: str):
#         sio.leave_room(sid, topic)

#     def get(self, topic: str):
#         return self.topics.get(topic, None)

#     async def publish(self, topic: str, message):
#         timestamp = datetime.now().timestamp()
#         self.topics[topic] = {"message": message, "timestamp": timestamp}
#         await sio.emit("message-available", timestamp, room=topic)

# topicManager = TopicManager()

# use context storage: within the aiohttp app, see 
# class TopicData(TypedDict):
#     message: Any
#     timestamp: float

# topic_buffers: Dict[str, TopicData] = {}

topics: Dict[str, Dict] = {}
app["topics"] = topics
app["problem"] = {"fitProblem": None, "filepath": None}
problem = None

async def index(request):
    """Serve the client-side application."""
    # redirect to static built site:
    return web.HTTPFound('/static/index.html')
    
@sio.event
async def connect(sid, environ):
    print("connect ", sid)

@sio.event
async def load_model_file(sid: str, pathlist: List[str], filename: str):
    from bumps.cli import load_model
    path = Path(*pathlist, filename)
    app["problem"]["fitProblem"] = load_model(str(path))
    await sio.emit("plot_update_ready", True)

def get_single_probe_data(theory, probe, substrate=None, surface=None, label=''):
    fresnel_calculator = probe.fresnel(substrate, surface)
    Q, FQ = probe.apply_beam(probe.calc_Q, fresnel_calculator(probe.calc_Q))
    Q, R = theory
    assert isinstance(FQ, np.ndarray)
    if len(Q) != len(probe.Q):
        # Saving interpolated data
        output = dict(Q = Q, theory = R, fresnel=np.interp(Q, probe.Q, FQ))
    elif getattr(probe, 'R', None) is not None:
        output = dict(Q = probe.Q, dQ = probe.dQ, R = probe.R, dR = probe.dR, theory = R, fresnel = FQ)
    else:
        output = dict(Q = probe.Q, dQ = probe.dQ, theory = R, fresnel = FQ)
    output['label'] = label
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
def get_plot_data(sid: str, view: str = 'linear'):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    result = []
    for model in fitProblem.models:
        theory = model.reflectivity()
        probe = model.probe
        result.append(get_probe_data(theory, probe, model._substrate, model._surface))
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
    return to_dict(result)

@sio.event
def get_profile_plot(sid: str):
    import mpld3
    import matplotlib.pyplot as plt
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    fig = plt.figure()
    for model in iter(fitProblem.models):
        model.plot_profile()
    return mpld3.fig_to_dict(plt.gcf())

@sio.event
def get_parameters(sid: str):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    if fitProblem is None:
        return []
    raw_parameters = fitProblem._parameters
    parameters = []
    for rp in raw_parameters:
        p = {}
        low, high = (v for v in rp.prior.limits)
        p['min'] = VALUE_FORMAT.format(nice(low))
        p['max'] = VALUE_FORMAT.format(nice(high))
        p['value_str'] = VALUE_FORMAT.format(nice(rp.value))
        p['value01'] = rp.prior.get01(rp.value)
        p['name'] = rp.name
        parameters.append(p)
    return to_dict(parameters)

@sio.event
async def set_parameter01(sid: str, parameter_name: str, parameter_value01: float):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    if fitProblem is None:
        return
    parameter_matches = [p for p in fitProblem._parameters if p.name == parameter_name]
    if len(parameter_matches) < 1:
        return
    parameter = parameter_matches[0]
    new_value  = parameter.prior.put01(parameter_value01)
    nice_new_value = nice(new_value, digits=VALUE_PRECISION)
    parameter.value = nice_new_value
    fitProblem.model_update()
    await sio.emit("plot_update_ready", True)
    return new_value

@sio.event
async def subscribe(sid: str, topic: str):
    sio.enter_room(sid, topic)
    last_timestamp = topics.get(topic, {}).get("timestamp", None)
    if last_timestamp is not None:
        await sio.emit(topic, last_timestamp, to=sid)

@sio.event
def unsubscribe(sid: str, topic: str):
    sio.leave_room(sid, topic)

@sio.event
async def publish(sid: str, topic: str, message):
    timestamp = datetime.now().timestamp()
    topics[topic] = {"message": message, "timestamp": timestamp}
    await sio.emit(topic, timestamp, room=topic)

@sio.event
def get_last_message(sid: str, topic: str):
    # this is a GET request in disguise -
    # emitter must handle the response in a callback,
    # as no separate response event is emitted.  
    return topics.get(topic, {}).get("message", None)

@sio.event
def get_dirlisting(sid: str, pathlist: List[str]):
    # GET request
    # TODO: use psutil to get disk listing as well?
    subfolders = []
    files = []
    for p in Path(*pathlist).iterdir():
        if p.is_dir():
            subfolders.append(p.name)
        else:
            files.append(p.resolve().name)
    return dict(subfolders=subfolders, files=files)

app.on_startup.append(lambda App: publish('', 'fitter_defaults', FITTER_DEFAULTS))
app.on_startup.append(lambda App: publish('', 'fitter_settings', deepcopy(FITTER_DEFAULTS)))
app.on_startup.append(lambda App: publish('', 'fitter_active', 'amoeba'))

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

app.router.add_static('/static', static_dir_path)
app.router.add_get('/', index)

VALUE_PRECISION = 6
VALUE_FORMAT = "{{:.{:d}g}}".format(VALUE_PRECISION)

def nice(v, digits=4):
    """Fix v to a value with a given number of digits of precision"""
    from math import log10, floor
    if v == 0. or not np.isfinite(v):
        return v
    else:
        sign = v/abs(v)
        place = floor(log10(abs(v)))
        scale = 10**(place-(digits-1))
        return sign*floor(abs(v)/scale+0.5)*scale
    

if __name__ == '__main__':
    web.run_app(app)
