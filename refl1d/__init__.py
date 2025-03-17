# This program is in the public domain
# Author: Paul Kienzle
"""
Refl1D: Specular Reflectometry Modeling and Fitting Software

This package provides tools for modeling a variety of systems from
simple slabs to complex systems with a mixture models involving
smoothly varying layers.

X-ray and neutron and polarized neutron data can be loaded and the model
parameters adjusted to achieve the best fit.

A graphical interface allows direct manipulation of the model profiles.

See http://refl1d.readthedocs.org for online manuals.
"""

import os
import sys
from typing import Literal

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"

__schema_version__ = "2"

BACKEND_NAMES = Literal["numba", "c_ext", "python"]
BACKEND_NAME: BACKEND_NAMES = os.environ.get("REFL1D_BACKEND", "numba")


def use(backend_name: BACKEND_NAMES):
    global BACKEND_NAME
    BACKEND_NAME = backend_name
    if "refl1d.backends" in sys.modules:
        # then it's already been imported:
        import refl1d.backends as backends

        backends.set_backend(backend_name)
