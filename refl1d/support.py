"""
Support files for the application.

This includes tools to help with testing, documentation, command line
parsing, etc. which are specific to this application, rather than general
utilities.
"""

import os
from pathlib import Path


def get_data_path():
    """
    Locate the examples directory.
    """
    # Check for data path in the environment
    key = "REFL1D_DATA"
    if key in os.environ:
        path = os.environ[key]
        if not os.path.isdir(path):
            raise RuntimeError("Path in environment %s not a directory" % key)
        return path

    # Check for data next to the package.
    try:
        root = Path(__file__).resolve().parent.parent.parent
        return root / "doc" / "examples"
    except Exception:
        raise RuntimeError("Could not find sample data")


_REGISTRY = {
    # List the example datasets that are needed for demos and doctests.
    # Each line looks like:
    #
    #     'dataset': ['subdirectory', 'filename'],
    #
    "spin_valve01.refl": ["spinvalve", "spin_valve01.refl"],
    "chale207.refl": ["polymer", "10ndt001.refl"],
    "10ndt001.refl": ["polymer", "10ndt001.refl"],
    "lha03_255G.refl": ["spinvalve", "n101Gc1.refl"],
}


def sample_data(file):
    examples = get_data_path()
    if file in _REGISTRY:
        return os.path.join(examples, *_REGISTRY[file])
    else:
        raise ValueError("Sample dataset %s not available" % file)
