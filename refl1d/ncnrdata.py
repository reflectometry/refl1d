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

import numpy
from numpy import inf, pi

from .resolution import Monochromatic
from . import util

def load(filename, instrument=None, **kw):
    """
    Return a probe for NCNR data.
    """
    content = parse_file(filename)
    return _make_probe(geometry=Monochromatic(), content=content, **kw)

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

def _make_probe(geometry, header, data, **kw):
    header.update(**kw)
    Q,R,dR = data
    resolution = geometry.resolution(Q, **header)
    probe = resolution.probe(data=(R,dR))
    probe.title = header['title']
    probe.date = header['date']
    probe.instrument = header['instrument']
    return probe

class NCNRLoader:
    def load(self, filename, **kw):
        header, data = parse_file(filename)
        header.update(**kw)
        return _make_probe(geometry=self, header=header, data=data, **kw)

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
