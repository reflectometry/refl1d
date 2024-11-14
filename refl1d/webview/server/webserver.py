from dataclasses import dataclass
from pathlib import Path

import bumps.cli
from bumps.webview.server import webserver
from bumps.webview.server.webserver import (
    main,
)

# Register the refl1d model loader
import refl1d.fitplugin

bumps.cli.install_plugin(refl1d.fitplugin)


webserver.CDN_TEMPLATE = "https://cdn.jsdelivr.net/npm/refl1d-webview-client@{client_version}/dist/{client_version}"
webserver.CLIENT_PATH = Path(__file__).parent.parent / "client"


@dataclass
class Refl1DOptions(webserver.BumpsOptions):
    serializer: webserver.SERIALIZERS = "dataclass"
    headless: bool = True


webserver.OPTIONS_CLASS = Refl1DOptions
webserver.APPLICATION_NAME = "refl1d"

if __name__ == "__main__":
    main()
