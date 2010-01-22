# This program is in the public domain
# Author: Paul Kienzle
"""
NCNR data loaders.

The following instruments are defined::

    Ng1, ANDR, NG7 and Xray

These are :class:`resolution.Monochromatic` classes tuned with 
default instrument parameters and loaders for reduced NCNR data.
See :module:`resolution` for details.
"""

import numpy
from numpy import inf, pi

from .resolution import Monochromatic

def load(filename, instrument=None, **kw):
    """
    Return a probe for NCNR data.
    """
    content = parse_ncnr_file(filename)
    return _make_probe(geometry=Monochromatic(), content=content, **kw)

def parse_ncnr_file(filename):
    """
    Parse NCNR reduced data.

    Returns a dictionary with fields for e.g., date, title, and 
    instrument.  The columns field is a list of column names, and
    the data field is an array containing the data.
    
    Slit geometry is set to the default from the instrument if it is not
    available in the reduced file.
    """
    file = open(filename, 'r')

    # Parse header
    content = {}
    for line in file:
        words = line.split()
        field,value = parse_line(line)
        content[field] = value
        if field == 'columns': break        
    content['columns'] = content['columns'].split()
        
    # Parse data
    content['data'] = numpy.loadtxt(file).T

    # Fill in instrument parameters, if not available from the file
    instrument = INSTRUMENT[content['instrument']]
    content.setdefault('radiation',instrument.radiation)
    content.setdefault('wavelength',instrument.wavelength)
    content.setdefault('dLoL',instrument.wavelength)
    content.setdefault('d_s1',instrument.d_s1)
    content.setdefault('d_s2',instrument.d_s2)
    
    return content

def _make_probe(geometry, content, **kw):
    content.update(**kw)
    Q,R,dR = content['data']
    resolution = geometry.resolution(Q, **content)
    probe = resolution.probe(data=(R,dR))
    probe.title = content['title']
    probe.date = content['date']
    probe.instrument = content['instrument']
    return probe

class NCNRLoader:
    def load(self, filename, **kw):
        content = parse_ncnr_file(filename)
        content.update(**kw)
        return _make_probe(geometry=self, content=content, **kw)

class ANDR(Monochromatic,NCNRLoader):
    """
    Instrument definition for NCNR AND/R diffractometer/reflectometer.
    """
    instrument = "AND/R"
    radiation = "neutron"
    wavelength = 5.0042
    dLoL=0.009
    d_s1 = 230.0 + 1856.0
    d_s2 = 230.0

class NG1(Monochromatic,NCNRLoader):
    """
    Instrument definition for NCNR NG-1 reflectometer.
    """
    instrument = "NG-1"
    radiation = "neutron"
    wavelength = 4.75
    dLoL=0.015
    d_s1 = 75*2.54
    d_s2 = 14*2.54
    d_s3 = 9*2.54
    d_s4 = 42*2.54

class NG7(Monochromatic,NCNRLoader):
    """
    Instrument definition for NCNR NG-7 reflectometer.
    """
    instrument = "NG-7"
    radiation = "neutron"
    wavelength = 4.768
    dLoL=0.040
    d_s1 = None
    d_s2 = None
    d_detector = 2000

class XRay(Monochromatic,NCNRLoader):
    """
    Instrument definition for NCNR X-ray reflectometer.

    Normal dT is in the range 2e-5 to 3e-4.

    Slits are fixed throughout the experiment in one of a
    few preconfigured openings.  Please update this file with
    the standard configurations when you find them.

    You can choose to ignore the geometric calculation entirely
    by setting the slit opening to 0 and using sample_broadening
    to define the entire divergence::

        xray = ncnrdata.Xray(slits_at_To=0)
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

