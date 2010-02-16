# This program is in the public domain
# Author: Paul Kienzle
"""
NCNR data loaders.

The following instruments are defined::

    ANDR, NG1, NG7 and Xray

These are :class:`resolution.Monochromatic` classes tuned with
default instrument parameters and loaders for reduced NCNR data.
See :module:`resolution` for details.
"""

import os
import numpy
from numpy import inf, pi

from .resolution import Monochromatic
from . import util

def load(filename, instrument=None, **kw):
    """
    Return a probe for NCNR data.
    """
    if filename is None: return None
    if instrument is None: instrument=Monochromatic()
    header,data = parse_file(filename)
    header.update(**kw)
    Q,R,dR = data
    resolution = instrument.resolution(Q, **header)
    probe = resolution.probe(data=(R,dR))
    probe.title = header['title']
    probe.date = header['date']
    probe.instrument = header['instrument']
    probe.filename = filename
    return probe

def load_magnetic(filename, **kw):
    """
    Return a probe for magnetic NCNR data.
    
    *Tguide* is the guide angle.
    """
    Tguide = kw.pop('Tguide',270)
    probes = [load(v, **kw) for v in find_xsec(filename)]
    return PolarizedNeutronProbe(probes, Tguide=Tguide)

def find_xsec(filename):
    """
    Find files containing the polarization cross-sections.
    
    Returns tuple with file names for ++ +- -+ -- cross sections, or
    None if the spin cross section does not exist.

    # TODO: check whether I have the spin states correct
    'a' corresponds to spin --
    'b' corresponds to spin -+
    'c' corresponds to spin +-
    'd' corresponds to spin ++

    Unfortunately the interpretation is a little more complicated than
    this as the data acquisition system assigns letter on the basis of
    flipper state rather than neutron spin state.  Whether flipper on
    or off corresponds to spin up or down depends on whether the
    polarizer/analyzer is a supermirror in transmission or reflection
    mode, or in the case of ^3He polarizers, whether the polarization
    is up or down.
    """
    if filename[-1] in 'abcdABCD':
        filename = filename[:-1]
    def check(a):
        if os.path.exists(filename+a): return filename+a
        elif os.path.exists(filename+a.upper()): return filename+a.upper()
        else: return None
    return (check('d'),check('c'),check('b'),check('a'))

def parse_file(filename):
    """
    Parse NCNR reduced data file returning *header* and *data*.

    *header* dictionary of fields such as 'data', 'title', 'instrument'
    *data* 2D array of data

    If 'columns' is present in header, it will be a list of the names of
    the columns.  If 'instrument' is present in the header, the default
    instrument geometry will be specified.

    Slit geometry is set to the default from the instrument if it is not
    available in the reduced file.
    """
    header, data = util.parse_file(filename)

    # Fill in instrument parameters, if not available from the file
    if 'instrument' in header and header['instrument'] in INSTRUMENTS:
        instrument = INSTRUMENTS[header['instrument']]
        header.setdefault('radiation',instrument.radiation)
        header.setdefault('wavelength',str(instrument.wavelength))
        header.setdefault('dLoL',str(instrument.dLoL))
        header.setdefault('d_s1',str(instrument.d_s1))
        header.setdefault('d_s2',str(instrument.d_s2))

    if 'columns' in header: header['columns'] = header['columns'].split()
    for key in ('wavelength','dLoL','d_s1','d_s2'):
        if key in header: header[key] = float(header[key])

    return header, data

class NCNRLoader:
    def load(self, filename, **kw):
        return load(filename, instrument=self)
    def load_magnetic(self, filename, **kw):
        pass

class ANDR(Monochromatic, NCNRLoader):
    """
    Instrument definition for NCNR AND/R diffractometer/reflectometer.
    """
    instrument = "AND/R"
    radiation = "neutron"
    wavelength = 5.0042
    dLoL = 0.009
    d_s1 = 230.0 + 1856.0
    d_s2 = 230.0

class NG1(Monochromatic, NCNRLoader):
    """
    Instrument definition for NCNR NG-1 reflectometer.
    """
    instrument = "NG-1"
    radiation = "neutron"
    wavelength = 4.75
    dLoL = 0.015
    d_s1 = 75*25.4
    d_s2 = 14*25.4
    d_s3 = 9*25.4
    d_s4 = 42*25.4

class NG7(Monochromatic, NCNRLoader):
    """
    Instrument definition for NCNR NG-7 reflectometer.
    """
    instrument = "NG-7"
    radiation = "neutron"
    wavelength = 4.768
    dLoL = 0.040
    d_s1 = None
    d_s2 = None
    d_detector = 2000

class XRay(Monochromatic, NCNRLoader):
    """
    Instrument definition for NCNR X-ray reflectometer.

    Normal dT is in the range 2e-5 to 3e-4.

    Slits are fixed throughout the experiment in one of a
    few preconfigured openings.  Please update this file with
    the standard configurations when you find them.

    You can choose to ignore the geometric calculation entirely
    by setting the slit opening to 0 and using sample_broadening
    to define the entire divergence::

        xray = ncnrdata.XRay(slits_at_To=0)
        data = xray.load("exp123.dat", sample_broadening=1e-4)
    """
    instrument = "X-ray"
    radiation = "xray"
    wavelength = 1.5416
    dLoL = 1e-3/wavelength
    d_s1 = 275.5
    d_s2 = 192.5
    d_s3 = 175.0
    d_detector = None

# Instrument names assigned by reflpak
INSTRUMENTS = {
    'CG-1': ANDR,
    'NG-1': NG1,
    'NG-7': NG7,
    'Xray': XRay,
    }
