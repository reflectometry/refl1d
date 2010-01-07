# This program is in the public domain
# Author: Paul Kienzle
"""
NCNR data loaders.

Usage is as follows::

    instrument = ncnrdata.INSTRUMENT(key=value, ...)
    probe = instrument.load(filename, key=value, ...)

where INSTRUMENT is one of NG1, ANDR, NG7, Xray or Generic.

The properties specific to the experiment are::

    *Tlo* (degrees):  Start of opening slit region, or inf if none
    *Thi* (degrees):  End of opening slit region, or inf if none
    *slits_at_Tlo* (mm) or (mm,mm):  Slit 1 and slit 2 openings at Tlo

For example, to load two datasets measured with the same slits::

    instrument = ncnrdata.ANDR(slits_at_Tlo=0.21, Tlo=0.35)
    data1 = instrument.load('bse004.refl')
    data2 = instrument.load('bse012.refl')

Sample properties may be given if available::

    *sample_width* (mm): use this when sample is tiny
    *sample_broadening* (degrees): use this when sample is warped

When the sample is tiny, stray neutrons miss the sample and are not
reflected onto the detector.  This results in a resolution that is
tighter than expected given the slit openings.

When the sample is warped, it make act to either focus or spread the
incident beam.  Some samples are diffuse scatters, which also acts
to spread the beam.  The degree of spread can be estimated from the
full-width at half max (FWHM) of a rocking curve at known slit settings.
The expected FWHM will be (s1+s2) / (2*(d_s1-d_s2)).  The difference
between this and the measured FWHM is the sample_broadening value.

The resolution can also be computed for an experiment simulation:

    instrument = ncnrdata.INSTRUMENT(key=value, ...)
    probe = instrument.probe(A=vector, key=value, ...)

where *A* is a vector of angles such as numpy.linspace(0,5,100).  For
instruments which scan in Q (e.g., NG-7), use *Q* instead of *A*.

With class Generic, arbitrary reflectometry data can be loaded or
simulated, but you will need to specify some instrument parameters::

    *L* (Angstroms):  instrument wavelength
    *dLoL*:           wavelength dispersion
    *d_s1* (mm):      distance from sample to slit 1
    *d_s2* (mm):      distance from sample to slit 2

For polychromatic instruments, wavelength L will be a vector. Wavelength
dispersion dLoL may also be a vector if it is not the same for each wavelength.

To show the default values for an instrument use::

    print ncnrdata.defaults(ncnrdata.INSTRUMENT)

This works both for the predefined instrument values and for the
specific values for a measurement.
"""

import numpy
from numpy import inf, pi

from . import probe, resolution

class Generic:
    """
    Data loader for generic scanning reflectometers.
    """
    L = None
    dLoL = None
    d_s1 = None
    d_s2 = None
    Tlo= inf
    Thi= inf
    slits_at_Tlo = (0,0)
    slits_below = None
    slits_above = None
    sample_width = 1e10
    sample_broadening = 0

    def __init__(self, **kw):
        for k,v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unexpected keyword argument '%s'"%k)
            setattr(self, k, v)

    def load(self, filename, **kw):
        """
        Load the data, returning the associated probe.  This probe will
        contain Q, angle, wavelength, measured reflectivity and the
        associated uncertainties.

        You can override any generic instrument parameters for this particular
        file using key=value.
        """
        # Load the data
        data = numpy.loadtxt(filename).T
        if data.shape[0] == 2:
            Q,R = data
            dR = None
        elif data.shape[0] > 2:
            Q,R,dR = data[:3]
            # ignore extra columns like dQ or L
        return self._calc(Q=Q, data=(R,dR), **kw)

    def probe(self, A=None, Q=None, **kw):
        """
        Generate the probe associated with a measurement.  This probe will
        contain, Q, angle, wavelength and the associated uncertainties,
        but not any data. Only one of Q or angle A should be specified.

        You can override any generic instrument parameters for this particular
        file using key=value.
        """
        if (Q is None) != (A is None):
            raise ValueError("probe needs one of angle or q")
        if Q is None: # Compute Q from angle A and wavelength L
            L = kw.pop('L',self.L)
            dLoL = kw.pop('dLoL',self.dLoL)
            if not isscalar(A) and not isscalar(L):
                # Support multiple wavelengths at multiple angles.
                nA = len(A)
                nL = len(L)
                L = numpy.repeat(L,nA)
                if not isscalar(dLoL):
                    dLoL = numpy.repeat(dLoL,nA)
                A = numpy.tile(A,nL)
            Q = 4*pi/L*sin(radians(A))
            return self._calc(Q=Q, L=L, dLoL=dLoL, data=None, **kw)
        else:
            return self._calc(Q=Q, data=None, **kw)

    def _calc(self, Q=None, data=None, **kw):
        """
        Perform the resolution calculation.
        """
        # Get measurement properties
        L = kw.pop('L',self.L)
        dLoL = kw.pop('dLoL',self.dLoL)
        Tlo = kw.pop('Tlo',self.Tlo)
        Thi = kw.pop('Thi',self.Thi)
        slits_at_Tlo = kw.pop('slits_at_Tlo',self.slits_at_Tlo)
        slits_below = kw.pop('slits_below',self.slits_below)
        slits_above = kw.pop('slits_above',self.slits_above)
        d_s1 = kw.pop('d_s1',self.d_s1)
        d_s2 = kw.pop('d_s2',self.d_s2)
        sample_width = kw.pop('sample_width',self.sample_width)
        sample_broadening = kw.pop('sample_broadening',self.sample_broadening)

        # Compute the angular divergence in radians
        T = resolution.QL2T(Q=Q,L=L)
        slits = resolution.slits_from_angle(T=T, slits_at_Tlo=slits_at_Tlo,
                                            Tlo=Tlo, Thi=Thi,
                                            slits_below=slits_below,
                                            slits_above=slits_above)
        dT = resolution.divergence(T=T,
                                   slits=slits,
                                   sample_width=sample_width,
                                   d_s1 = d_s1, d_s2 = d_s2)
        dT += sample_broadening

        # Generate a probe object with the data for evaluating the model
        if self.radiation is "neutron":
            return probe.NeutronProbe(A=T,dA=dT,L=L,dL=dLoL*L,data=data)
        else:
            return probe.XrayProbe(A=T,dA=dT,L=L,dL=dLoL*L,data=data)

    def __str__(self):
        msg = """\
== Instrument %(name)s ==
radiation = %(radiation)s at %(L)g +/- %(dLpercent)g%% Angstrom
slit distances = %(d_s1)g mm and %(d_s2)g mm
fixed region below %(Tlo)g and above %(Thi)g degrees
slit openings at Tlo are %(slits_at_Tlo)s mm
sample width = %(sample_width)g mm
sample broadening = %(sample_broadening)g degrees
""" % dict(name=self.instrument, L=self.L, dLpercent=self.dLoL*100,
           d_s1=self.d_s1, d_s2=self.d_s2,
           sample_width=self.sample_width, Tlo=self.Tlo, Thi=self.Thi,
           slits_at_Tlo=str(self.slits_at_Tlo), radiation=self.radiation,
           )
        return msg

    @classmethod
    def defaults(cls):
        """
        Return default instrument properties as a printable string.
        """
        msg = """\
== Instrument %(name)s ==
radiation = %(radiation)s at %(L)g +/- %(dLpercent)g%% Angstrom
slit distances = %(d_s1)g mm and %(d_s2)g mm
""" % dict(name=cls.instrument, L=cls.L, dLpercent=cls.dLoL*100,
           d_s1=cls.d_s1, d_s2=cls.d_s2,
           radiation=cls.radiation,
           )
        return msg

class ANDR(Generic):
    """
    Loader for reduced data from the NCNR AND/R instrument.
    """
    instrument = "AND/R"
    radiation = "neutron"
    L = 5.0042
    dLoL=0.009
    d_s1 = 230.0 + 1856.0
    d_s2 = 230.0

class NG1(Generic):
    """
    Loader for reduced data from the NCNR NG-1 instrument.
    """
    instrument = "NG-1"
    radiation = "neutron"
    L = 4.75
    dLoL=0.015
    d_s1 = 75*2.54
    d_s2 = 14*2.54
    d_s3 = 9*2.54
    d_s4 = 42*2.54

class NG7(Generic):
    """
    Loader for reduced data from the NCNR NG-7 instrument.
    """
    instrument = "NG-7"
    radiation = "neutron"
    L = 4.768
    dLoL=0.040
    d_s1 = None
    d_s2 = None
    d_detector = 2000

class XRay(Generic):
    """
    Loader for reduced data from the NCNR X-ray instrument.

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
    L = 1.5416
    dLoL = 1e-3*L
    d_s1 = 275.5
    d_s2 = 192.5
    d_s3 = 175.0
    d_detector = None
