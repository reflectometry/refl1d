"""
***Warning***: importing cli modifies the behaviour of bumps
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

from bumps.webview.server import cli
from bumps.webview.server.cli import BumpsOptions

from . import api  # uses side-effects to register refl1d functions
from refl1d import __version__

# Register the refl1d model loader
# and the serialized model migrations
from refl1d.bumps_interface import fitplugin
from bumps.cli import install_plugin

install_plugin(fitplugin)

CLIENT_PATH = Path(__file__).parent.parent / "client"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "align":
        # Command line tool to regenerate the profile uncertainty plot:
        #
        #   refl1d align <model>.py <store> [<layer>.<offset>] [0|1|2|n]
        #
        from refl1d.uncertainty import run_errors

        del sys.argv[1]
        run_errors()
    else:
        cli.plugin_main(name="refl1d", client=CLIENT_PATH, version=__version__)


def start_refl1d_server(options: Optional[BumpsOptions] = None):
    """
    Start a Jupyter server for the webview.
    This returns an asyncio.Task object that should be awaited
    to ensure the server starts without exceptions.
    """
    from bumps.webview.server import api
    from bumps.webview.server.webserver import start_app

    api.state.app_name = "refl1d"
    api.state.app_version = __version__
    api.state.client_path = CLIENT_PATH

    return asyncio.create_task(start_app(options, jupyter_link=True))


if __name__ == "__main__":
    main()
