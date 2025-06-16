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


def refl1d_server():
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

    return asyncio.create_task(start_app(jupyter_link=True))


if __name__ == "__main__":
    main()
