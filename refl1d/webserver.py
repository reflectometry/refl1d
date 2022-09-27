# from .main import setup_bumps

from aiohttp import web
import socketio
from pathlib import Path
import json
from copy import deepcopy

from bumps.fitters import DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit

FITTERS = (DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit)
FITTER_DEFAULTS = {}
for fitter in FITTERS:
    FITTER_DEFAULTS[fitter.id] = {
        "name": fitter.name,
        "settings": dict(fitter.settings)
    }

# set up mutable state of fitters:
fitter_settings = deepcopy(FITTER_DEFAULTS)
active_fitter = "amoeba"

sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
static_dir_path = Path(__file__).parent / 'webview'
sio.attach(app)

async def index(request):
    """Serve the client-side application."""
    # redirect to static built site:
    return web.HTTPFound('/static/index.html')
    
@sio.event
async def connect(sid, environ):
    print("connect ", sid)
    sio.enter_room(sid, "situation")
    await sio.emit("fitter-defaults", FITTER_DEFAULTS, to=sid)
    await sio.emit("fitter-settings", fitter_settings, to=sid)
    await sio.emit("fitter-active", active_fitter, to=sid)

@sio.on("fitter-active")
async def new_fitter_active(sid, data):
    # using global state here:
    global active_fitter
    active_fitter = data
    await sio.emit("fitter-active", data, room="situation")

@sio.on("fitter-settings")
async def new_fitter_settings(sid, data):
    global fitter_settings
    fitter_settings.update(data)
    await sio.emit("fitter-settings", fitter_settings, room="situation")

# async def get_sessions(sid):
#     return web.Response(text=json.dumps(list(sessions)), content_type='application/json')

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

app.router.add_static('/static', static_dir_path)
# app.router.add_static('/static', os.path.join(currdir, 'webview'))
app.router.add_get('/', index)


if __name__ == '__main__':
    web.run_app(app)
