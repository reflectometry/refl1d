# This program is in the public domain
# Author: Paul Kienzle
"""
SNS data loaders.

The following instruments are defined::

    Liquids, Magnetic

These are :class:`resolution.Polychromatic` classes tuned with
default instrument parameters and loaders for reduced SNS data.
See :module:`resolution` for details.
"""

import re
import numpy
from .resolution import Polychromatic, binwidths
from . import util

def load(filename, instrument=None, **kw):
    """
    Return a probe for NCNR data.
    """
    header, data = parse_file(filename)
    return _make_probe(geometry=Polychromatic(), header=header, data=data, **kw)

def _make_probe(geometry, header, data, **kw):
    header.update(**kw)
    Q,dQ,R,dR,L = data
    dL = binwidths(L)
    T = kw.pop('angle',util.QL2T(Q[0],L[0]))
    resolution = geometry.resolution(L=L, dL=dL, T=T, **header)
    probe = resolution.probe(data=(R,dR))
    probe.title = header['title']
    probe.date = header['date']
    probe.instrument = header['instrument']
    return probe

def parse_file(filename):
    """
    Parse SNS reduced data, returning *header* and *data*.

    *header* dictionary of fields such as 'data', 'title', 'instrument'
    *data* 2D array of data
    """
    raw_header, data = util.parse_file(filename)
    header = {}

    # guess instrument from file name
    original_file = raw_header.get('F','unknown')
    if 'REF_L' in original_file:
        instrument = 'Liquids'
    elif 'REF_M' in original_file:
        instrument = 'Magnetic'
    else:
        instrument = 'unknown'
    header['instrument'] = instrument
    header['filename'] = original_file
    header['radiation'] = 'neutron'

    # Plug in default instrument values for slits
    if 'instrument' in header and header['instrument'] in INSTRUMENTS:
        instrument = INSTRUMENTS[header['instrument']]
        header['d_s1'] = instrument.d_s1
        header['d_s2'] = instrument.d_s2

    # Date-time field for the file
    header['date'] = raw_header.get('D','')
    
    # Column names and units
    columnpat = re.compile(r'(?P<name>\w+)[(](?P<units>\w*)[)]')
    columns,units = zip(*columnpat.findall(raw_header.get('L','')))
    header['columns'] = columns
    header['units'] = units
    
    # extra information like title, angle, etc.
    commentpat = re.compile(r'(?P<name>.*)\s*:\s*(?P<value>.*)\s*\n')
    comments = dict(commentpat.findall(raw_header.get('C','')))
    header['title'] = comments.get('Title','')
    header['description'] = comments.get('Notes','')
    
    # parse values of the form "Long Name: (value, 'units')" in comments
    valuepat = re.compile(r"[(]\s*(?P<value>.*)\s*,\s*'(?P<units>.*)'\s*[)]")
    def parse_value(valstr):
        d = valuepat.match(valstr).groupdict()
        return float(d['value']),d['units']
    if 'Detector Angle' in comments:
        header['angle'],_ = parse_value(comments['Detector Angle'])

    return header, data


class SNSLoader:
    def load(self, filename, **kw):
        header,data = parse_file(filename)
        return _make_probe(geometry=self, header=header, data=data, **kw)

print "Insert correct slit distances for Liquids and Magnetic"
class Liquids(Polychromatic, SNSLoader):
    """
    Loader for reduced data from the SNS Liquids instrument.
    """
    instrument = "Liquids"
    radiation = "neutron"
    wavelength = 2.5,17.5
    dLoL = 0.02
    d_s1 = 230.0 + 1856.0
    d_s2 = 230.0

class Magnetic(Polychromatic, SNSLoader):
    """
    Loader for reduced data from the SNS Magnetic instrument.
    """
    instrument = "Magnetic"
    radiation = "neutron"
    wavelength = 1.8,14
    dLoL = 0.02
    d_s1 = 75*2.54
    d_s2 = 14*2.54

# Instrument names assigned by reflpak
INSTRUMENTS = {
    'Liquids': Liquids,
    'Magnetic': Magnetic,
    }



# ===== utils ==============

def feather(L,counts=100000,range=None):
    """
    Return expected intensity as a function of wavelength given the TOF
    feather range and the total number of counts.

    TOF feather is approximately a boltzmann distribution with gaussian
    convolution.  The following looks pretty enough; don't know how well it
    corresponds to the actual SNS feather.
    """
    import scipy.stats
    y = numpy.linspace(-4,4,10)
    G = numpy.exp(-y**2/10)
    x = numpy.arange(12,85)
    B = scipy.stats.boltzmann.pmf(x, 0.05, 1, loc=16)
    BGz = numpy.convolve(B,G,mode='same')
    if range is None: range = L[0],L[-1]
    if range[0] > range[1]: range = range[::-1]
    range = range[0]*(1-1e-15),range[1]*(1+1e-15)
    z = numpy.linspace(range[0],range[1],len(BGz))
    pL = numpy.interp(L,z,BGz,left=0,right=0)
    nL = pL/sum(pL)*counts
    return  nL
