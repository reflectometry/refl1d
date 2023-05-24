import asyncio
from aiohttp import web
from dataclasses import dataclass
import json
from pathlib import Path
import socket
from typing import Union, Dict, List, Optional

import numpy as np
from bumps.webview.server import webserver

# Register the refl1d model loader
import refl1d.fitplugin
import bumps.cli
bumps.cli.install_plugin(refl1d.fitplugin)

from . import api # use side-effects to register refl1d functions
from .profile_plot import plot_sld_profile_plotly

webserver.CDN_TEMPLATE = "https://cdn.jsdelivr.net/npm/refl1d-webview-client@{client_version}/dist"
webserver.CLIENT_PATH = Path(__file__).parent.parent / 'client'

@dataclass
class Refl1DOptions(webserver.BumpsOptions):
    serializer: webserver.SERIALIZERS = "dataclass"
    headless: bool = True

webserver.OPTIONS_CLASS = Refl1DOptions

main = webserver.main

if __name__ == '__main__':
    main()