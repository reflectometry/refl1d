"""
Generate api docs for all modules in a package.

Drop this file in your sphinx doc directory, and change the constants at
the head of this file as appropriate.  Make sure this file is on the python
path and add the following to the end of conf.py::

    import genmods
    genmods.make()

OPTIONS are the options for gen_api_files().

PACKAGE is the dotted import name for the package.

MODULES is the list fo modules to include in table of contents order.

PACKAGE_TEMPLATE is the template for the api index file.

MODULE_TEMPLATE is the template for each api module.
"""

import inspect
import sys
from os import makedirs
from os.path import abspath, dirname, exists, getmtime
from os.path import join as joinpath

OPTIONS = {
    "absolute": False,  # True if package.module in table of contents
    "dir": "api",  # Destination directory for the api docs
    "root": "../refl1d",  # Source directory for the package, or None for default
}

PACKAGE = "refl1d"

MODULES = [
    # ('__init__', 'Top level namespace'),
    # ('interface', 'Interface'),
    # ('deprecated.magnetic', 'Magnetic Models'),
    # ("names", "Public API"),
    ("deprecated.freeform", "Freeform - Parametric B-Spline"),
    ("backends", "Backends for low level reflectivity calculations"),
    ("models.bumps_interface.fitplugin", "Bumps plugin definition for reflectivity models"),
    ("models.bumps_interface.fitproblem", "Bumps FitProblem definition for reflectivity models"),
    ("models.dist", "Non-uniform samples"),
    ("models.experiment", "Reflectivity fitness function"),
    ("models.probe.abeles", "Pure python reflectivity calculator"),
    ("models.probe.data_loaders.anstodata", "Reader for ANSTO data format"),
    ("models.probe.data_loaders.ncnrdata", "NCNR Data"),
    ("models.probe.data_loaders.snsdata", "SNS Data"),
    ("models.probe.data_loaders.staj", "Staj File"),
    ("models.probe.data_loaders.stajconvert", "Staj File Converter"),
    ("models.probe.fresnel", "Pure python Fresnel reflectivity calculator"),
    ("models.probe.instrument", "Reflectivity instrument definition"),
    ("probe", "Instrument probe"),
    ("models.probe.resolution", "Beam resolution calculations"),
    ("models.probe.stitch", "Overlapping reflectivity curve stitching"),
    ("models.profile", "Model profile"),
    ("models.sample.cheby", "Freeform - Chebyshev model"),
    ("models.sample.flayer", "Functional layers"),
    ("models.sample.layers", "Models of Reflectivity Layer Structure"),
    ("models.sample.magnetism", "Magnetic Models"),
    ("models.sample.material", "Material"),
    ("models.sample.materialdb", "Materials Database"),
    ("models.sample.mono", "Freeform - Monotonic Spline"),
    ("models.sample.polymer", "Polymer models"),
    ("models.sample.reflectivity", "Basic reflectivity calculations"),
    ("models.uncertainty", "Plot sample profile uncertainty"),
    ("utils.support", "Environment support"),
    ("utils.util", "Miscellaneous functions"),
]


PACKAGE_TEMPLATE = """.. Autogenerated by genmods.py -- DO NOT EDIT --

.. _%(package)s-index:

#########
Reference
#########

.. only:: html

   :Release: |version|
   :Date: |today|

.. toctree::
   :hidden:

   %(rsts)s

.. currentmodule:: %(package)s

.. autosummary::

   %(mods)s

"""

MODULE_TEMPLATE = """.. Autogenerated by genmods.py -- DO NOT EDIT --

%(sep)s
%(prefix)s%(module)s - %(title)s
%(sep)s

.. currentmodule:: %(package)s.%(module)s

.. autosummary::
   :nosignatures:

   %(members)s

.. automodule:: %(package)s.%(module)s
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

"""
# ===================== Documentation generator =====================


def newer(file1, file2):
    return not exists(file1) or (getmtime(file1) < getmtime(file2))


def get_members(package, module):
    name = package + "." + module
    try:
        __import__(name)
    except Exception:
        print("while importing " + name)
        raise
    M = sys.modules[name]
    try:
        L = M.__all__
    except AttributeError:
        L = [s for s in sorted(dir(M)) if inspect.getmodule(getattr(M, s)) == M and not s.startswith("_")]
    return L


def gen_api_docs(package, modules, dir="api", absolute=True, root=None):
    """
    Generate .rst files in *dir* from *modules* in *package*.

    *dir* defaults to 'api'

    *absolute* is True if modules are listed as package.module in the table
    of contents.  Default is True.

    *root* is the path to the package source.  This may be different from
    the location of the package in the python path if the documentation is
    extracted from the build directory rather than the source directory.
    The source is used to check if the module definition has changed since
    the rst file was built.
    """

    # Get path to package source
    if root is None:
        __import__(package)
        M = sys.modules[package]
        root = abspath(dirname(M.__file__))

    # Build path to documentation tree
    prefix = package + "." if absolute else ""
    if not exists(dir):
        makedirs(dir)

    # Update any modules that are out of date.  Compiled modules
    # will always be updated since we only check for .py files.
    for module, title in modules:
        modfile = joinpath(root, module + ".py")
        rstfile = joinpath(dir, module + ".rst")
        sep = "*" * len(f"{prefix}{module} - {title}")
        if not exists(modfile):
            __import__(package + "." + module)
            modfile = sys.modules[package + "." + module].__file__
        if newer(rstfile, modfile):
            members = "\n    ".join(get_members(package, module))
            # print("writing %s"%rstfile)
            with open(rstfile, "w") as f:
                f.write(MODULE_TEMPLATE % locals())

    # Update the table of contents, but only if the configuration
    # file containing the module list has changed.  For now, that
    # is the current file.
    api_index = joinpath(dir, "index.rst")
    if newer(api_index, __file__):
        rsts = "\n   ".join(module + ".rst" for module, _ in modules)
        mods = "\n   ".join(prefix + module for module, _ in modules)
        # print("writing %s"%api_index)
        with open(api_index, "w") as f:
            f.write(PACKAGE_TEMPLATE % locals())


def make():
    gen_api_docs(PACKAGE, MODULES, **OPTIONS)
