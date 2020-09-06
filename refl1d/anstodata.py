# This program is in the public domain
# Author: Andrew Nelson
"""
ANSTO data loaders

The following instrument is defined::

    Platypus

All the ANSTO instruments emit Q/R/dR/dQ in their output files.
"""

import re
import os.path
import numpy as np
from .instrument import Pulsed
from bumps.data import maybe_open
from .probe import QProbe, PolarizedNeutronProbe
from .resolution import FWHM2sigma


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

    # it would be nice to use refl1d.probe.load4 directly. However,
    # there are lots of files in the wild that don't use # to denote comments
    # in the header. In addition, it is known that ANSTO datasets emit dQ as
    # FWHM.

    with maybe_open(f) as fh:
        fname = fh.name
        header_lines = 0
        for i, line in enumerate(fh):
            try:
                nums = [float(tok) for tok in re.split(r'\s|,', line)
                        if len(tok)]
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

def load_magnetic(pp=None, pm=None, mp=None, mm=None, Aguide=270, H=0, instrument=None, **kw):
    """
    Return a probe for ANSTO polarised neutron data.

    **Parameters**

    pp :    ++ spin cross-section. Default: 'None'
    pm :    +- spin cross-section. Default: 'None'
    mp :    -+ spin cross-section. Default: 'None'
    mm :    -- spin cross-section. Default: 'None'

                filenames for all spin cross section datasets. If not all cross-sections are
                measured, put 'None'. 
                e.g. 
                
                >>> instrument = Platypus()
                >>> probe = instrument.load_magnetic(["PLP0045001.dat", None, None, "PLP0045003.dat"])

    **Returns**

    probe : probe.QProbe
    """
    probes = [ load(f) for f in [pp, pm, mp, mm] if f is not None ]

    if all(p is None for p in probes):
        raise IOError("Data set has no magnetic cross sections: %r" % filename)
    probe = PolarizedNeutronProbe(probes, Aguide=Aguide, H=H)
    if shared_beam:
        probe.shared_beam()  # Share the beam parameters by default
    return probe



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

    def load_magnetic(self, filename, **kw):
        return load_magnetic(filename, instrument=self, **kw)


class Platypus(ANSTOData, Pulsed):
    """
    Loader for reduced data from the ANSTO Platypus instrument.
    """
    instrument = "Platypus"
    radiation = "neutron"
    wavelength = (2.5,12.5) #typical wavelength range for polarised measurements
    d_s1 = 290.0 + 2844.8 # mm
    d_s2 = 290.0 # mm
    dLoL = 0.043 # Ranges from 0.018 - 0.09 depending on chopper settings

INSTRUMENTS = {
    'Platypus': Platypus,
    }
