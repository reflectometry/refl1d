# import asyncio
# from aiohttp import web
from dataclasses import dataclass

# import json
from pathlib import Path
# import socket
# from typing import Union, Dict, List, Optional

# import numpy as np
from bumps.webview.server import webserver
from bumps.webview.server.webserver import (
    # start_app,
    # sio,
    main,
    # create_server_task,
    # display_inline_jupyter,
    # open_tab_link,
)

# Register the refl1d model loader
from refl1d.bumps_interface import fitplugin
import bumps.cli

bumps.cli.install_plugin(fitplugin)

from . import api  # use side-effects to register refl1d functions
# from .profile_plot import plot_sld_profile_plotly

webserver.CLIENT_PATH = Path(__file__).parent.parent / "client"


@dataclass
class Refl1DOptions(webserver.BumpsOptions):
    serializer: webserver.SERIALIZERS = "dataclass"
    headless: bool = True


webserver.OPTIONS_CLASS = Refl1DOptions
webserver.APPLICATION_NAME = "refl1d"

if __name__ == "__main__":
    main()
