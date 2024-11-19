# Authors: Paul Kienzle, Brian Maranville
"""
Reflectometry backend loader
"""

import importlib

from . import BACKEND_NAME, BACKEND_NAMES

BACKEND_MODULE_NAMES = {
    "c_ext": "refl1d.reflmodule",
    "numba": "refl1d.lib.numba",
    "python": "refl1d.lib.python",
}

backend = None


def set_backend(backend_name: BACKEND_NAMES):
    global backend
    backend_module_name = BACKEND_MODULE_NAMES.get(backend_name, None)
    if backend_module_name is None:
        raise ValueError(f"Unknown backend: {backend_name}")
    backend = importlib.import_module(backend_module_name)


if BACKEND_NAME is not None:
    set_backend(BACKEND_NAME)
