import asyncio
from aiohttp import web
from dataclasses import dataclass
import json
from pathlib import Path
import socket
from typing import Union, Dict, List, Optional

import numpy as np
from bumps.webview.server import webserver
from bumps.webview.server.webserver import (
    app, setup_sio_api, get_commandline_options
)

# Register the refl1d model loader
import refl1d.fitplugin
import bumps.cli
bumps.cli.install_plugin(refl1d.fitplugin)

from . import api # use side-effects to register refl1d functions
from .profile_plot import plot_sld_profile_plotly

client_path = Path(__file__).parent.parent / 'client'
index_path = client_path / 'dist'
static_assets_path = index_path / 'assets'

@dataclass
class Options(webserver.Options):
    serializer: webserver.SERIALIZERS = "dataclass"
    headless: bool = True
    app_name: str = "Refl1D"


async def index(request):
    """Serve the client-side application."""
    # check if the locally-build site has the correct version:
    with open(client_path / 'package.json', 'r') as package_json:
        client_version = json.load(package_json)['version'].strip()

    try:
        local_version = open(index_path / 'VERSION', 'rt').read().strip()
    except FileNotFoundError:
        local_version = None

    print(index_path, local_version, client_version, local_version == client_version)
    if client_version == local_version:
        return web.FileResponse(index_path / 'index.html')
    else:
        CDN = f"https://cdn.jsdelivr.net/npm/refl1d-webview-client@{client_version}/dist"
        with open(client_path / 'index_template.txt', 'r') as index_template:
            index_html = index_template.read().format(cdn=CDN)
        return web.Response(body=index_html, content_type="text/html")

def main(options: Optional[Options] = None, sock: Optional[socket.socket] = None):
    options = get_commandline_options(arg_defaults={"serializer": "dataclass", "headless": False}) if options is None else options
    options.app_name = "Refl1D"
    try:
        asyncio.run(start_app(options, sock))
    except KeyboardInterrupt:
        print("stopped by KeyboardInterrupt.")

async def start_app(options: Options = Options(), sock: Optional[socket.socket] = None):
    setup_sio_api()
    runsock = webserver.setup_app(options=options, static_assets_path=static_assets_path, index=index, sock=sock)
    await web._run_app(app, sock=runsock)

if __name__ == '__main__':
    main()