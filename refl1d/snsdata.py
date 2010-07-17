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
import math
import numpy
from .resolution import Polychromatic, binwidths
from . import util

## Estimated intensity vs. wavelength for liquids reflectometer
LIQUIDS_FEATHER = numpy.array([
  (0.821168,28.5669),
  (0.930657,23.5032),
  (1.04015,19.0127),
  (1.14964,16.9108),
  (1.25912,15.4777),
  (1.45073,15.6688),
  (1.61496,16.6242),
  (1.83394,18.4395),
  (2.02555,20.6369),
  (2.29927,23.6943),
  (2.57299,23.6943),
  (2.87409,21.1146),
  (3.22993,15.5732),
  (3.58577,12.8981),
  (4.07847,9.4586),
  (4.5438,6.59236),
  (5.11861,4.68153),
  (5.7208,3.05732),
  (6.37774,1.91083),
  (7.19891,1.24204),
  (8.04745,0.955414),
  (9.06022,0.573248),
  (10.1825,0.477707),
  (11.4142,0.382166),
  (12.8102,0.191083),
  (14.3431,0.286624),
]).T


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
    probe = resolution.probe(data=(R,dR), **header)
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

    def simdata(self, sample, counts=None, **kw):
        """
        Simulate a run with a particular sample.
        """
        from reflectometry.reduction.rebin import rebin
        from .experiment import Experiment
        from .resolution import binedges
        T = kw.pop('T', self.T)
        slits = kw.pop('slits', self.slits)
        if slits is None: slits = [0.2*Ti for Ti in T]
        if counts is None: counts = [(100*Ti)**4 for Ti in T]

        # Compute reflectivity with resolution and added noise
        probes = []
        for Ti,Si,Ci in zip(T,slits,counts):
            probe = self.simulate(T=Ti, slits=Si, **kw)
            M = Experiment(probe=probe, sample=sample)
            I = rebin(binedges(self.feather[0]),self.feather[1],
                      binedges(probe.L))
            I /= sum(I)
            _, Rth = M.reflectivity()
            Rcounts = numpy.random.poisson(Rth*I*Ci)
            Icounts = I*Ci
            # Z = X/Y
            # var Z = (var X / X**2 + var Y / Y**2) * Z**2
            #       = (1/X + 1/Y) * (X/Y)**2
            #       = (Y + X) * X/Y**3
            R = Rcounts/Icounts
            dR = numpy.sqrt((Icounts + Rcounts)*Rcounts/Icounts**3)
            probe.data = R,dR
            probes.append(probe)

        return probes

# TODO: print "Insert correct slit distances for Liquids and Magnetic"
class Liquids(Polychromatic, SNSLoader):
    """
    Loader for reduced data from the SNS Liquids instrument.
    """
    instrument = "Liquids"
    radiation = "neutron"
    feather = LIQUIDS_FEATHER
    wavelength = 1.5,5.
    #wavelength = 0.5,5
    #wavelength = 5.5,10
    #wavelength = 10.5,15
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

def intensity_from_spline(Lrange,dLoL,feather):
    from danse.reflectometry.reduction import rebin
    L0,L1 = Lrange
    n = math.ceil(math.log(L1/L0)/math.log(1+dLoL))
    L = L0*(1+dLoL)**numpy.arange(0,n)
    return (L[:-1]+L[1:])/2, rebin(feather[0],feather[1],L)
    

def boltzmann_feather(L,counts=100000,range=None):
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
    #if range is None: range = L[0],L[-1]
    #if range[0] > range[1]: range = range[::-1]
    #range = range[0]*(1-1e-15),range[1]*(1+1e-15)
    #z = numpy.linspace(range[0],range[1],len(BGz))
    z = numpy.linspace(2,16.5,len(BGz))  # Wavelength range for liquids
    pL = numpy.interp(L,z,BGz,left=0,right=0)
    nL = pL/sum(pL)*counts
    return  nL
