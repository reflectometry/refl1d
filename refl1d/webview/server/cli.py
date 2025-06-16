"""
***Warning***: importing cli modifies the behaviour of bumps
"""

import asyncio
from pathlib import Path

from bumps.webview.server import cli

from . import api  # uses side-effects to register refl1d functions
from refl1d import __version__

# Register the refl1d model loader
# and the serialized model migrations
from refl1d.bumps_interface import fitplugin
from bumps.cli import install_plugin

install_plugin(fitplugin)

CLIENT_PATH = Path(__file__).parent.parent / "client"


def main():
    cli.plugin_main(name="refl1d", client=CLIENT_PATH, version=__version__)


def start_jupyter_server():
    """
    Start a Jupyter server for the webview.
    """
    from bumps.webview.server import api
    from bumps.webview.server.cli import BumpsOptions
    from bumps.webview.server.webserver import start_app

    api.state.app_name = "refl1d"
    api.state.app_version = __version__
    api.state.client_path = CLIENT_PATH
    options = BumpsOptions()

    return asyncio.create_task(start_app(options, jupyter_link=True))


if __name__ == "__main__":
    main()
