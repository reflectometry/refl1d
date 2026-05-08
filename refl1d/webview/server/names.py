# from bumps.api import state, set_problem, load_session
# from .webserver import main, start_app, create_server_task, sio

try:
    from bumps.names import set_problem, load_session
    from bumps.api import state
    from bumps.webview.webserver import start_app, sio
except ImportError: # CRUFT: bumps 1.1 rearranged internal structure
    from bumps.webview.server.api import state, set_problem, load_session
    from bumps.webview.server.webserver import start_app, sio

from .cli import main
