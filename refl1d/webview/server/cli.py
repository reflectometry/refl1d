"""
***Warning***: importing cli modifies the behaviour of bumps
"""

from pathlib import Path

from bumps.webview.server import cli

from . import api  # uses side-effects to register refl1d functions

# Register the refl1d model loader
# from refl1d.bumps_interface import fitplugin
# from .profile_plot import plot_sld_profile_plotly

CLIENT_PATH = Path(__file__).parent.parent / "client"

# from dataclasses import dataclass
# from bumps.webview.server.cli import BumpsOptions, SERIALIZERS
# @dataclass
# class Refl1DOptions(BumpsOptions):
#    serializer: SERIALIZERS = "dataclass"
#    headless: bool = True
# bumps_cli.OPTIONS_CLASS = Refl1DOptions


def main():
    cli.plugin_main(name="refl1d", client=CLIENT_PATH)


if __name__ == "__main__":
    main()
