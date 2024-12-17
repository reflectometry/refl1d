# This program is in the public domain
# Author: Andrew Nelson
"""
ANSTO data loaders

The following instrument is defined::

    Platypus

All the ANSTO instruments emit Q/R/dR/dQ in their output files.
"""

import os.path
import re

import numpy as np
from bumps.data import maybe_open

from ..probe import QProbe
from ..resolution import FWHM2sigma


def _load_dat(f):
    """
    Loads a Platypus dataset from file. This will normally be Q, R, dR, dQ.

    | Q - Momentum transfer |1/Ang|
    | R - Reflectivity
    | dR - uncertainty in reflectivity (1 sigma)
    | dQ - FWHM of Gaussian resolution kernel.

    **Parameters**

    f : file-handle or string
        File to load the dataset from.

    **Returns**

    (filename, name, data) - str, str, tuple of np.ndarray
    """

    # it would be nice to use refl1d.probe.data_loaders.load4 directly. However,
    # there are lots of files in the wild that don't use # to denote comments
    # in the header. In addition, it is known that ANSTO datasets emit dQ as
    # FWHM.

    with maybe_open(f) as fh:
        fname = fh.name
        header_lines = 0
        for i, line in enumerate(fh):
            try:
                nums = [float(tok) for tok in re.split(r"\s|,", line) if len(tok)]
            except ValueError:
                continue
            if len(nums) >= 2:
                header_lines = i
                break

    data = np.loadtxt(f, unpack=True, skiprows=header_lines)

    filename = fname
    name = os.path.splitext(os.path.basename(fname))[0]

    return filename, name, data


def load(filename, instrument=None, **kw):
    """
    Return a probe for ANSTO data.

    **Parameters**

    f : file-handle or string
        File to load the dataset from.

    **Returns**

    probe : probe.QProbe
    """
    fname, name, data = _load_dat(filename)

    Q = data[0]
    R = data[1]
    dR = data[2]
    dQ = FWHM2sigma(data[3])

    probe = QProbe(Q, dQ, data=(R, dR), name=name, filename=fname)

    if instrument is not None:
        probe.instrument = instrument.instrument

    return probe


class ANSTOData(object):
    def load(self, filename, **kw):
        return load(filename, instrument=self, **kw)


class Platypus(ANSTOData):
    """
    Loader for reduced data from the ANSTO Platypus instrument.
    """

    instrument = "Platypus"
    radiation = "neutron"


INSTRUMENTS = {
    "Platypus": Platypus,
}
