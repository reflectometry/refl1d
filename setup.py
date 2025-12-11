#!/usr/bin/env python
import sys
import os
from setuptools import setup, Extension

BUILD_EXTENSION = bool(os.environ.get("BUILD_EXTENSION", False)) or "build_ext" in sys.argv


# reflmodule extension
def reflmodule_config():
    S = (
        "reflmodule.cc",
        "methods.cc",
        "reflectivity.cc",
        "magnetic.cc",
        "contract_profile.cc",
        "convolve.cc",
        "convolve_sampled.cc",
        "build_profile.cc",
    )

    Sdeps = ("erf.cc", "methods.h", "rebin.h", "rebin2D.h", "reflcalc.h")
    sources = [os.path.join("refl1d", "lib", "c", f) for f in S]
    depends = [os.path.join("refl1d", "lib", "c", f) for f in Sdeps]
    return Extension(
        "refl1d.reflmodule",
        sources=sources,
        depends=depends,
        language="c++",
    )


dist = setup(
    data_files=[("share/icons", ["extra/refl1d-icon.svg"])],
    ext_modules=[reflmodule_config()] if BUILD_EXTENSION else [],
)
# End of file
