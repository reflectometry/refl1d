"""
***Warning***: importing cli modifies the behaviour of bumps
"""

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


if __name__ == "__main__":
    main()
