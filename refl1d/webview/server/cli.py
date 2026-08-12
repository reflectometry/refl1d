"""
Refl1D entry points.

*main()* starts bumps with the Refl1D plugin when *refl1d* is given on the command line.

*start_refl1d_server()* starts bumps with the Refl1D plugin from jupyter notebooks.

***Warning***: importing refl1d.webview.server modifies the behaviour of bumps
"""

import sys
import asyncio
from pathlib import Path

from . import api  # uses side-effects to register refl1d functions
from refl1d import __version__

# Register the refl1d model loader
# and the serialized model migrations
from refl1d.bumps_interface import fitplugin

try:
    from bumps.plugin import install_plugin
    from bumps.cli import plugin_main
    from bumps.webview.webserver import start_app
except ImportError:
    # CRUFT: bumps < 1.1
    from bumps.cli import install_plugin
    from bumps.webview.server.cli import plugin_main
    from bumps.webview.server.webserver import start_app

install_plugin(fitplugin)

CLIENT_PATH = Path(__file__).parent.parent / "client"


def main():
    """
    Run refl1d from the command line.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "align":
        # Command line tool to regenerate the profile uncertainty plot:
        #
        #   refl1d align <model>.py <store> [<layer>.<offset>] [0|1|2|n]
        #
        from refl1d.uncertainty import run_errors

        del sys.argv[1]
        run_errors()
    else:
        plugin_main(name="refl1d", client=CLIENT_PATH, version=__version__)


def start_refl1d_server():
    """
    Start a Jupyter server for the webview.
    This returns an asyncio.Task object that should be awaited
    to ensure the server starts without exceptions.
    """
    api.state.app_name = "refl1d"
    api.state.app_version = __version__
    api.state.client_path = CLIENT_PATH

    return asyncio.create_task(start_app(jupyter_link=True))


if __name__ == "__main__":
    main()
