# This program is in the public domain
# Author: Paul Kienzle
"""
Reflectometry instrument definitions.

Given Q = 4 pi sin(theta)/lambda, the instrumental resolution in Q is
determined by the dispersion angle theta and wavelength lambda.  For
monochromatic instruments, the wavelength resolution is fixed and the
angular resolution varies.  For polychromatic instruments, the wavelength
resolution varies and the angular resolution is fixed.

The angular resolution is determined by the geometry (slit positions and
openings and sample profile) with perhaps an additional contribution
from sample warp.  For monochromatic instruments, measurements are taken
with fixed slits at low angles until the beam falls completely onto the
sample.  Then as the angle increases, slits are opened to preserve full
illumination.  At some point the slit openings exceed the beam width,
and thus they are left fixed for all angles above this threshold.

When the sample is tiny, stray neutrons miss the sample and are not
reflected onto the detector.  This results in a resolution that is
tighter than expected given the slit openings.  If the sample width
is available, we can use that to determine how much of the beam is
intercepted by the sample, which we then use as an alternative second
slit.  This simple calculation isn't quite correct for very low Q, but
data in this region will be contaminated by the direct beam, so we
won't be using those points.

When the sample is warped, it may act to either focus or spread the
incident beam.  Some samples are diffuse scatters, which also acts
to spread the beam.  The degree of spread can be estimated from the
full-width at half max (FWHM) of a rocking curve at known slit settings.
The expected FWHM will be (s1+s2) / (2*(d_s1-d_s2)).  The difference
between this and the measured FWHM is the sample_broadening value.
A second order effect is that at low angles the warping will cast
shadows, changing the resolution and intensity in very complex ways.

For polychromatic time of flight instruments, the wavelength dispersion
is determined by the reduction process which usually bins the time
channels in a way that sets a fixed relative resolution dL/L for each bin.

Usage
=====

:module:`instrument` (this module) defines two instrument types:
:class:`Monochromatic` and :class:`Polychromatic`.  These represent
generic scanning and time of flight instruments, respectively.

To perform a simulation or load a data set, an instrument must first
be defined.  For example:

    >>> from instrument import Monochromatic
    >>> andr = Monochromatic(
            # instrument parameters
            instrument = "AND/R",
            radiation = "neutron",
            wavelength = 5.0042,
            dLoL=0.009,
            d_s1 = 230.0 + 1856.0,
            d_s2 = 230.0,
            # measurement parameters
            Tlo=0.5, slits_at_Tlo=0.2, slits_below=0.1,
            )

Fixed parameters for various instruments are defined in :module:`snsdata`
and :module:`ncnrdata`, so the above is equivalent to:

    >>> from ncnrdata import ANDR
    >>> andr = ANDR(Tlo=0.5, slits_at_Tlo=0.2, slits_below=0.1)

With a particular measurement geometry defined by the instrument we can
compute the expected resolution function for arbitrary angles:

    >>> T,dT,L,dL = andr.resolution(T=linspace(0,5,51))
    >>> Q,dQ = dTdL2dQ(T=T,dT=dT,L=L,dL=dL)

More commonly, though, the instrument would be used to generate a
measurement probe for use in modeling or to read in a previously
measured data set:

    >>> simulation_probe = andr.probe(T=linspace(0,5,51))
    >>> measured_probe = andr.load('blg117.refl')

For magnetic systems a polarized beam probe is needed::

    >>> probe = andr.magnetic_probe(T=numpy.arange(0,5,100))

When loading or simulating a data set, any of the instrument parameters
and measurement geometry information can be specified, replacing the
defaults within the instrument.  For example, to include sample broadening
effects in the resolution::

    >>> probe1 = andr.load('blg117.refl', sample_broadening=0.1)

Properties of the instrument can be displayed, both for the generic
instrument (which defines slit distances and wavelength in this case)
or for the specific measurement geometry (which adds detail about the
slit opening as a function of angle)::

    >>> print ANDR.defaults()
    >>> print andr.defaults()

GUI Usage
=========

Graphical user interfaces follow different usage patterns from scripts.
Here the emphasis will be on selecting a data set to process, displaying
its default metadata and allowing the user to override it.

File loading should follow the pattern established in reflectometry
reduction, with an extension registry and a fallback scheme whereby
files can be checked in a predefined order.  If the file cannot be
loaded, then the next loader is tried.  This should be extended with
the concept of a magic signature such as those used by graphics and
sound file applications: preread the first block and run it through
the signature check before trying to load it.  For unrecognized
extensions, all loaders can be tried.

The file loader should return an instrument instance with metadata
initialized from the file header.  This metadata can be displayed
to the user along with a plot of the data and the resolution.  When
metadata values are changed, the resolution can be recomputed and the
display updated.  When the data set is accepted, the final resolution
calculation can be performed.

Algorithms
==========

Resolution in Q is computed from uncertainty in wavelength L and angle T
using propagation of errors::

    dQ**2 = (df/dL)**2 dL**2 + (df/dT)**2 dT**2

where::

    f(L,T) = 4 pi sin(T)/L
    df/dL = -4 pi sin(T)/L**2 = -Q/L
    df/dT = 4 pi cos(T)/L = cos(T) Q/sin(T) = Q/tan(T)

yielding the traditional form::

    (dQ/Q)**2 = (dL/L)**2 + (dT/tan(T))**2

Computationally, 1/tan(T) is infinity at T=0, so it is better to use the
direct calculation::

    dQ = (4 pi / L) sqrt( sin(T)**2 (dL/L)**2 + cos(T)**2 dT**2 )


Wavelength dispersion dL/L is usually constant (e.g., for AND/R it is 2%
FWHM), but it can vary on time-of-flight instruments depending on how the
data is binned.

Angular divergence dT comes primarily from the slit geometry, but can have
broadening or focusing due to a warped sample.  The FWHM divergence due
to slits is::

    dT_slits = degrees(0.5*(s1+s2)/(d1-d2))

where s1,s2 are slit openings edge to edge and d1,d2 are the distances
between the sample and the slits.  For tiny samples of width m, the sample
itself can act as a slit.  If s = sin(T)*m is smaller than s2 for some T,
then use::

    dT_slits = degrees(0.5*(s1+sin(T)*m)/d1)

The sample broadening can be read off a rocking curve using::

    dT_sample = w - dT_slits

where w is the measured FWHM of the peak in degrees and dT_slits is the
slit contribution to the divergence. Broadening can be negative for concave
samples which have a focusing effect on the beam.  This constant should
be added to the computed dT for all angles and slit geometries.  You will
not usually have this information on hand, but you can leave space for users
to enter it if it is available.

FWHM can be converted to 1-sigma resolution using the scale factor of
1/sqrt(8 * log(2))

For opening slits we assume dT/T is held constant, so if you know s and To
at the start of the opening slits region you can compute dT/To, and later
scale that to your particular T::

    dT(Q) = dT/To * T(Q)

Because d is fixed, that means s1(T) = s1(To) * T/To and s2(T) = s2(To) * T/To

"""

# TODO: the resolution calculator should not be responsible for loading
# the data; maybe do it as a mixin?

import numpy
from numpy import pi, inf, sqrt, log, degrees, radians, cos, sin, tan
from numpy import arcsin as asin, ceil
from numpy import ones_like, arange, isscalar, asarray
from util import TL2Q, QL2T, dTdL2dQ, dQdT2dLoL, FWHM2sigma, sigma2FWHM


class Monochromatic:
    """
    Instrument representation for scanning reflectometers.

    :Parameters:
        *instrument* : string
            name of the instrument
        *radiation* : string | xray or neutron
            source radiation type
        *d_s1*, *d_s2* : float | mm
            distance from sample to pre-sample slits 1 and 2; post-sample
            slits are ignored
        *wavelength* : float | Angstrom
            wavelength of the instrument
        *dLoL* : float
            constant relative wavelength dispersion; wavelength range and
            dispersion together determine the bins
        *slits_at_Tlo* : float OR (float,float) | mm
            slit 1 and slit 2 openings at Tlo; this can be a scalar if both
            slits are open by the same amount, otherwise it is a pair (s1,s2).
        *slits_at_Qlo* : float OR (float,float) | mm
            equivalent to slits_at_Tlo, for instruments that are controlled by
            Q rather than theta
        *Tlo*, *Thi* : float | degrees
            range of opening slits, or inf if slits are fixed.
        *Qlo*, *Qhi* : float | inv Angstroms
            range of opening slits when instrument is controlled by Q.
        *slits_below*, *slits_above* : float OR (float,float) | mm
            slit 1 and slit 2 openings below Tlo and above Thi; again, these
            can be scalar if slit 1 and slit 2 are the same, otherwise they
            are each a pair (s1,s2).  Below and above default to the values of
            the slits at Tlo and Thi respectively.
        *sample_width* : float | mm
            width of sample; at low angle with tiny samples, stray neutrons
            miss the sample and are not reflected onto the detector, so the
            sample itself acts as a slit, therefore the width of the sample
            may be needed to compute the resolution correctly
        *sample_broadening* : float | degrees FWHM
            amount of angular divergence (+) or focusing (-) introduced by
            the sample; this is caused by sample warp, and may be read off
            of the rocking curve by subtracting (s1+s2)/2/(d_s1-d_s2) from
            the FWHM width of the rocking curve
    """
    instrument = "monochromatic"
    radiation = "unknown"
    # Required attributes
    wavelength = None
    dLoL = None
    d_s1 = None
    d_s2 = None
    # Optional attributes
    Tlo= 90  # Use 90 for fixed slits.
    Thi= 90
    slits_at_Tlo = None    # Slit openings at Tlo, and default for slits_below
    slits_below = None     # Slit openings below Tlo, or fixed slits if Tlo=90
    slits_above = None
    sample_width = 1e10    # Large but finite value
    sample_broadening = 0

    def __init__(self, **kw):
        self._translate_Q_to_theta(kw)
        for k,v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise TypeError("unexpected keyword argument '%s'"%k)

    def load(self, filename, **kw):
        """
        Load the data, returning the associated probe.  This probe will
        contain Q, angle, wavelength, measured reflectivity and the
        associated uncertainties.

        You can override instrument parameters using key=value.  In
        particular, slit settings *slits_at_Tlo*, *Tlo*, *Thi*,
        and *slits_below*, and *slits_above* are used to define the
        angular divergence.

        Note that this function ignores any resolution information
        stored in the file, such as dQ, dT or dL columns, and instead
        uses the defined instrument parameters to calculate the resolution.
        """
        # Load the data
        data = numpy.loadtxt(filename).T
        if data.shape[0] == 2:
            Q,R = data
            dR = None
        elif data.shape[0] == 3:
            Q,R,dR = data
        elif data.shape[0] == 4:
            Q,dQ,R,dR = data
        elif data.shape[0] == 5:
            Q,dQ,R,dR,L = data
        kw["Q"] = Q
        self._translate_Q_to_theta(kw)
        T,dT,L,dL = self.resolution(**kw)
        del kw["T"]
        return make_probe(T=T,dT=dT,L=L,dL=dL,data=(R,dR),
                          radiation=self.radiation, **kw)

    def probe(self, **kw):
        """
        Return a probe for use in simulation.

        :Parameters:
            *Q* : [float] | Angstroms
                Q values to be measured.
            *T* : [float] | degrees
                Angles to be measured.

        Additional keyword parameters

        :Returns:
            *probe* : Probe
                Measurement probe with complete resolution information.  The
                probe will not have any data.

        If both *Q* and *T* are specified then *Q* takes precedents.

        You can override instrument parameters using key=value.  In
        particular, settings for *slits_at_Tlo*, *Tlo*, *Thi*,
        *slits_below*, and *slits_above* are used to define the
        angular divergence.
        """
        self._translate_Q_to_theta(kw)
        T,dT,L,dL = self.resolution(**kw)
        T = kw.pop('T')
        return make_probe(T=T,dT=dT,L=L,dL=dL,
                          radiation=self.radiation, **kw)

    def magnetic_probe(self, Tguide=270, shared_beam=True, **kw):
        """
        Simulate a polarized measurement probe.

        Returns a probe with Q, angle, wavelength and the associated
        uncertainties, but not any data.

        Guide field angle *Tguide* can be specified, as well as keyword
        arguments for the geometry of the probe cross sections such as
        *slits_at_Tlo*, *Tlo*, *Thi*, *slits_below*, and *slits_above*
        to define the angular divergence.
        """
        from .probe import PolarizedNeutronProbe
        probes = [self.probe(**kw) for _ in range(4)]
        probe = PolarizedNeutronProbe(probes, Tguide=Tguide)
        if shared_beam:
            probe.shared_beam()  # Share the beam parameters by default
        return probe

    def simulate(self, sample, uncertainty=0.01, **kw):
        """
        Simulate a run with a particular sample.

        :Parameters:
            *sample* : Stack
                Model of the sample.
            *uncertainty* = 0.01 : float
                Relative uncertainty in the measurement.

        Additional :meth:`probe` keyword parameters are required to define
        the set of angles to be measured

        :Returns:
            *experiment* : Experiment
                Sample + probe with simulated data.

        The relative uncertainty is used to calculate the number of incident
        beam intensity for the measurement as follows::

            I = (100 Q)^4 / s^2

        """
        from .experiment import Experiment
        probe = self.probe(**kw)
        M = Experiment(probe=probe, sample=sample)
        _, Rth = M.reflectivity()
        dR = uncertainty*M.fresnel()
        R = Rth + numpy.random.randn(*Rth.shape)*dR
        probe.data = R,dR

        return M

    def resolution(self, T, **kw):
        """
        Calculate resolution at each angle.

        :Parameters:
            *T* : [float] | degrees
                Q values for which resolution is needed

        :Return:
            *T*,*dT* : [float] | degrees
                Angles and angular divergence.
            *L*,*dL* : [float] | Angstroms
                Wavelengths and wavelength dispersion.
        """
        L = kw.get('L',kw.get('wavelength',self.wavelength))
        dLoL = kw.get('dLoL',self.dLoL)
        if L is None:
            raise TypeError("Need wavelength L to compute resolution")
        if dLoL is None:
            raise TypeError("Need wavelength dispersion dLoL to compute resolution")

        slits = self.calc_slits(T, **kw)
        dT = self.calc_dT(T, slits, **kw)

        return T,dT,L,dLoL*L

    def calc_slits(self, T, **kw):
        """
        Determines slit openings from measurement pattern.

        If slits are fixed simply return the same slits for every angle,
        otherwise use an opening range [Tlo,Thi] and the value of the
        slits at the start of the opening to define the slits.  Slits
        below Tlo and above Thi can be specified separately.

        *Tlo*,*Thi*      angle range over which slits are opening
        *slits_at_Tlo*   openings at the start of the range, or fixed opening
        *slits_below*, *slits_above*   openings below and above the range

        Use fixed_slits is available, otherwise use opening slits.
        """
        Tlo = kw.get('Tlo',self.Tlo)
        Thi = kw.get('Thi',self.Thi)
        slits_at_Tlo = kw.get('slits_at_Tlo',self.slits_at_Tlo)
        slits_below = kw.get('slits_below',self.slits_below)
        slits_above = kw.get('slits_above',self.slits_above)

        # Otherwise we are using opening slits
        if Tlo is None or slits_at_Tlo is None:
            raise TypeError("Resolution calculation requires Tlo and slits_at_Tlo")
        slits = slit_widths(T=T, slits_at_Tlo=slits_at_Tlo,
                            Tlo=Tlo, Thi=Thi,
                            slits_below=slits_below,
                            slits_above=slits_above)
        return slits

    def calc_dT(self, T, slits, **kw):
        """
        Compute the angular divergence for given slits and angles

        :Parameters:
            *T* : [float] | degrees
                measurement angles
            *slits* : float OR (float,float) | mm
                total slit opening from edge to edge, not beam center to edge
            *d_s1*, *d_s2* : float | mm
                distance from sample to slit 1 and slit 2
            *sample_width* : float | mm
                size of sample
            *sample_broadening* : float | degrees FWHM
                resolution changes from sample warp

        :Returns:
            *dT* : [float] | degrees FWHM
                angluar divergence

        *sample_broadening* can be estimated from W, the full width at half
        maximum of a rocking curve measured in degrees:

            sample_broadening = W - degrees( 0.5*(s1+s2) / (d1-d2))

        """
        d_s1 = kw.get('d_s1',self.d_s1)
        d_s2 = kw.get('d_s2',self.d_s2)
        if d_s1 is None or d_s2 is None:
            raise TypeError("Need slit distances d_s1, d_s2 to compute resolution")
        sample_width = kw.get('sample_width',self.sample_width)
        sample_broadening = kw.get('sample_broadening',self.sample_broadening)
        dT = divergence(T=T, slits=slits, distance=(d_s1,d_s2),
                        sample_width=sample_width,
                        sample_broadening=sample_broadening)

        return dT

    def _translate_Q_to_theta(self, kw):
        """
        Rewrite keyword arguments with Q values translated to theta values.
        """
        # Grab wavelength first so we can translate Qlo/Qhi to Tlo/Thi no
        # matter what order the keywords appear.
        wavelength = kw.get('wavelength',self.wavelength)
        if "Q" in kw:
            kw["T"] = QL2T(kw.pop("Q"), wavelength)
        if "Qlo" in kw:
            kw["Tlo"] = QL2T(kw.pop("Qlo"), wavelength)
        if "Qhi" in kw:
            kw["Thi"] = QL2T(kw.pop("Qhi"), wavelength)
        if "slits_at_Qlo" in kw:
            kw["slits_at_Tlo"] = kw.pop("slits_at_Qlo")

    def __str__(self):
        msg = """\
== Instrument %(name)s ==
radiation = %(radiation)s at %(L)g Angstrom with %(dLpercent)g%% resolution
slit distances = %(d_s1)g mm and %(d_s2)g mm
fixed region below %(Tlo)g and above %(Thi)g degrees
slit openings at Tlo are %(slits_at_Tlo)s mm
sample width = %(sample_width)g mm
sample broadening = %(sample_broadening)g degrees
""" % dict(name=self.instrument, L=self.wavelength, dLpercent=self.dLoL*100,
           d_s1=self.d_s1, d_s2=self.d_s2,
           sample_width=self.sample_width,
           sample_broadening=self.sample_broadening,
           Tlo=self.Tlo, Thi=self.Thi,
           slits_at_Tlo=str(self.slits_at_Tlo), radiation=self.radiation,
           )
        return msg

    @classmethod
    def defaults(cls):
        """
        Return default instrument properties as a printable string.
        """
        msg = """\
== Instrument class %(name)s ==
radiation = %(radiation)s at %(L)g Angstrom with %(dLpercent)g%% resolution
slit distances = %(d_s1)g mm and %(d_s2)g mm
""" % dict(name=cls.instrument, L=cls.wavelength, dLpercent=cls.dLoL*100,
           d_s1=cls.d_s1, d_s2=cls.d_s2,
           radiation=cls.radiation,
           )
        return msg

class Polychromatic:
    """
    Instrument representation for multi-wavelength reflectometers.

    :Parameters:
        *instrument* : string
            name of the instrument
        *radiation* : string | xray, neutron
            source radiation type
        *T* : float | degrees
            sample angle
        *slits* : float OR (float,float) | mm
            slit 1 and slit 2 openings
        *d_s1*, *d_s2* : float | mm
            distance from sample to pre-sample slits 1 and 2; post-sample
            slits are ignored
        *wavelength* : (float,float) | Angstrom
            wavelength range for the measurement
        *dLoL* : float
            constant relative wavelength dispersion; wavelength range and
            dispersion together determine the bins
        *sample_width* : float | mm
            width of sample; at low angle with tiny samples, stray neutrons
            miss the sample and are not reflected onto the detector, so the
            sample itself acts as a slit, therefore the width of the sample
            may be needed to compute the resolution correctly
        *sample_broadening* : float | degrees FWHM
            amount of angular divergence (+) or focusing (-) introduced by
            the sample; this is caused by sample warp, and may be read off
            of the rocking curve by subtracting 0.5*(s1+s2)/(d_s1-d_s2) from
            the FWHM width of the rocking curve
    """
    instrument = "polychromatic"
    radiation = "neutron" # unless someone knows how to do TOF Xray...
    # Required attributes
    d_s1 = None
    d_s2 = None
    slits = None
    T = None
    wavelength = None
    dLoL = None # usually 0.02 for 2% FWHM
    # Optional attributes
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

        You can override instrument parameters using key=value.
        In particular, slit settings *slits* and *T* define the
        angular divergence.
        """
        # Load the data
        data = numpy.loadtxt(filename).T
        Q,dQ,R,dR,L = data
        dL = binwidths(L)
        T = kw.pop('T',QL2T(Q,L))
        T,dT,L,dL = self.resolution(L=L, dL=dL, T=T, **kw)
        return make_probe(T=T,dT=dT,L=L,dL=dL,data=(R,dR),
                          radiation=self.radiation, **kw)

    def probe(self, **kw):
        """
        Simulate a measurement probe.

        Returns a probe with Q, angle, wavelength and the associated
        uncertainties, but not any data.

        You can override instrument parameters using key=value.
        In particular, slit settings *slits* and *T* define
        the angular divergence and *dLoL* defines the wavelength
        resolution.
        """
        low,high = kw.get('wavelength',self.wavelength)
        dLoL = kw.get('dLoL',self.dLoL)
        T = kw.pop('T',self.T)
        L = bins(low,high,dLoL)
        dL = binwidths(L)
        T,dT,L,dL = self.resolution(L=L, dL=dL, T=T, **kw)
        return make_probe(T=T,dT=dT,L=L,dL=dL,
                          radiation=self.radiation, **kw)

    def magnetic_probe(self, Tguide=270, shared_beam=True, **kw):
        """
        Simulate a polarized measurement probe.

        Returns a probe with Q, angle, wavelength and the associated
        uncertainties, but not any data.

        Guide field angle *Tguide* can be specified, as well as keyword
        arguments for the geometry of the probe cross sections such as
        slit settings *slits* and *T* to define the angular divergence
        and *dLoL* to define the wavelength resolution.
        """
        from .probe import PolarizedNeutronProbe
        probes = [self.probe(**kw) for _ in range(4)]
        probe = PolarizedNeutronProbe(probes, Tguide=Tguide)
        if shared_beam:
            probe.shared_beam()  # Share the beam parameters by default
        return probe

    def simulate(self, sample, uncertainty=0.01, **kw):
        """
        Simulate a run with a particular sample.

        :Parameters:
            *sample* : Stack
                Reflectometry model
            *T* : [float] | degrees
                List of angles to be measured, such as [0.15,0.4,1,2].
            *slits* : [float] or [(float,float)] | mm
                Slit settings for each angle. Default is 0.2*T
            *uncertainty* = 0.01 : float or [float]
                Incident intensity is set so that the worst dF/F is better
                than *uncertainty*, where F is the idealized Fresnel
                reflectivity of the sample.
            *dLoL* = 0.02: float
                Wavelength resolution
            *normalize* = True : boolean
                Whether to normalize the intensities
            *theta_offset* = 0 : float | degrees
                Sample alignment error
            *background* = 0 : float
                Background counts per incident neutron (background is
                assumed to be independent of measurement geometry).
            *back_reflectivity* = False : boolean
                Whether beam travels through incident medium
                or through substrate.
            *back_absorption* = 1 : float
                Absorption factor for beam traveling through substrate.
                Only needed for back reflectivity measurements.
        """
        from reflectometry.reduction.rebin import rebin
        from .experiment import Experiment
        from .instrument import binedges
        from .probe import ProbeSet
        T = kw.pop('T', self.T)
        slits = kw.pop('slits', self.slits)
        if slits is None: slits = [0.2*Ti for Ti in T]

        dLoL = kw.pop('dLoL', self.dLoL)
        normalize = kw.pop('normalize', True)
        theta_offset = kw.pop('theta_offset', 0)
        background = kw.pop('background', 0)
        back_reflectivity = kw.pop('back_reflectivity', False)
        back_absorption = kw.pop('back_absorption', 1)

        # Compute reflectivity with resolution and added noise
        probes = []
        for Ti,Si in zip(T,slits):
            probe = self.probe(T=Ti, slits=Si, dLoL=dLoL)
            probe.back_reflectivity = back_reflectivity
            probe.theta_offset.value = theta_offset
            probe.back_absorption.value = back_absorption
            M = Experiment(probe=probe, sample=sample)
            # Note: probe.L is reversed because L is sorted by increasing
            # Q in probe.
            I = rebin(binedges(self.feather[0]),self.feather[1],
                      binedges(probe.L[::-1]))[::-1]
            Ci = max(1./(uncertainty**2 * I * M.fresnel()))
            Icounts = Ci*I

            _, Rth = M.reflectivity()
            Rcounts = numpy.random.poisson(Rth*Icounts)
            if background > 0:
                Rcounts += numpy.random.poisson(Icounts*background,
                                                size=Rcounts.shape)
            # Set intensity/background _after_ calculating the theory function
            # since we don't want the theory function altered by them.
            probe.background.value = background
            # Correct for the feather.  This has to be done otherwise we
            # won't see the correct reflectivity.  Even if corrected for
            # the feather, though, we haven't necessarily corrected for
            # the overall number of counts in the measurement.
            # Z = X/Y
            # var Z = (var X / X**2 + var Y / Y**2) * Z**2
            #       = (1/X + 1/Y) * (X/Y)**2
            #       = (Y + X) * X/Y**3
            R = Rcounts/Icounts
            dR = numpy.sqrt((Icounts + Rcounts)*Rcounts/Icounts**3)

            if not normalize:
                #Ci = 1./max(R)
                R, dR = R*Ci, dR*Ci
                probe.background.value *= Ci
                probe.intensity.value = Ci

            probe.data = R,dR
            probes.append(probe)

        return Experiment(sample=sample, probe=ProbeSet(probes))

    def resolution(self, L, dL, **kw):
        """
        Return the resolution of the measurement.  Needs *T*, *L*, *dL*
        specified as keywords.
        """
        T = kw.pop('T', self.T)
        slits = kw.pop('slits', self.slits)
        dT = self.calc_dT(T,slits,**kw)

        # Compute the FWHM angular divergence in radians
        # Return the resolution
        return T,dT,L,dL

    def calc_dT(self, T, slits, **kw):
        d_s1 = kw.get('d_s1',self.d_s1)
        d_s2 = kw.get('d_s2',self.d_s2)
        sample_width = kw.get('sample_width',self.sample_width)
        sample_broadening = kw.get('sample_broadening',self.sample_broadening)
        dT = divergence(T=T, slits=slits, distance=(d_s1,d_s2),
                        sample_width=sample_width,
                        sample_broadening=sample_broadening)

        return dT

    def __str__(self):
        msg = """\
== Instrument %(name)s ==
radiation = %(radiation)s in %(L_min)g to %(L_max)g Angstrom with %(dLpercent)g%% resolution
slit distances = %(d_s1)g mm and %(d_s2)g mm
slit openings = %(slits)s mm
sample width = %(sample_width)g mm
sample broadening = %(sample_broadening)g degrees FWHM
""" % dict(name=self.instrument,
           L_min=self.wavelength[0], L_max=self.wavelength[1],
           dLpercent=self.dLoL*100,
           d_s1=self.d_s1, d_s2=self.d_s2, slits = str(self.slits),
           sample_width=self.sample_width,
           sample_broadening=self.sample_broadening,
           radiation=self.radiation,
           )
        return msg

    @classmethod
    def defaults(cls):
        """
        Return default instrument properties as a printable string.
        """
        msg = """\
== Instrument class %(name)s ==
radiation = %(radiation)s in %(L_min)g to %(L_max)g Angstrom with %(dLpercent)g%% resolution
slit distances = %(d_s1)g mm and %(d_s2)g mm
""" % dict(name=cls.instrument,
           L_min=cls.wavelength[0], L_max=cls.wavelength[1],
           dLpercent=cls.dLoL*100,
           d_s1=cls.d_s1, d_s2=cls.d_s2,
           radiation=cls.radiation,
           )
        return msg

def make_probe(**kw):
    """
    Return a reflectometry measurement object of the given resolution.
    """
    from .probe import NeutronProbe, XrayProbe, PROBE_KW
    radiation = kw.pop('radiation')
    kw = dict((k,v) for k,v in kw.items() if k in PROBE_KW)
    if radiation == 'neutron':
        return NeutronProbe(**kw)
    else:
        return XrayProbe(**kw)

def bins(low, high, dLoL):
    """
    Return bin centers from low to high perserving a fixed resolution.

    *low*,*high* are the minimum and maximum wavelength.
    *dLoL* is the desired resolution FWHM dL/L for the bins.
    """

    step = 1 + dLoL;
    n = ceil(log(high/low)/log(step))
    edges = low*step**arange(n+1)
    L = (edges[:-1]+edges[1:])/2
    return L

def binwidths(L):
    """
    Construct dL assuming that L represents the bin centers of a
    measured TOF data set, and dL is the bin width.

    The bins L are assumed to be spaced logarithmically with edges::

        E[0] = min wavelength
        E[i+1] = E[i] + dLoL*E[i]

    and centers::

        L[i] = (E[i]+E[i+1])/2
             = (E[i] + E[i]*(1+dLoL))/2
             = E[i]*(2 + dLoL)/2

    so::

        L[i+1]/L[i] = E[i+1]/E[i] = (1+dLoL)
        dL[i] = E[i+1]-E[i] = (1+dLoL)*E[i]-E[i]
              = dLoL*E[i] = 2*dLoL/(2+dLoL)*L[i]
    """
    if L[1] > L[0]:
        dLoL = L[1]/L[0] - 1
    else:
        dLoL = L[0]/L[1] - 1
    dL = 2*dLoL/(2+dLoL)*L
    return dL

def binedges(L):
    """
    Construct bin edges E assuming that L represents the bin centers of a
    measured TOF data set.

    The bins L are assumed to be spaced logarithmically with edges::

        E[0] = min wavelength
        E[i+1] = E[i] + dLoL*E[i]

    and centers::

        L[i] = (E[i]+E[i+1])/2
             = (E[i] + E[i]*(1+dLoL))/2
             = E[i]*(2 + dLoL)/2

    so::

        E[i] = L[i]*2/(2+dLoL)
        E[n+1] = L[n]*2/(2+dLoL)*(1+dLoL)
    """
    if L[1] > L[0]:
        dLoL = L[1]/L[0] - 1
        last = (1+dLoL)
    else:
        dLoL = L[0]/L[1] - 1
        last = 1./(1+dLoL)
    E = L*2/(2+dLoL)
    return numpy.hstack((E,E[-1]*last))

def divergence(T=None, slits=None, distance=None,
               sample_width=1e10, sample_broadening=0):
    """
    Calculate divergence due to slit and sample geometry.

    :Parameters:
        *T*         : float OR [float] | degrees
            incident angles
        *slits*     : float OR (float,float) | mm
            s1,s2 slit openings for slit 1 and slit 2
        *distance*  : (float,float) | mm
            d1,d2 distance from sample to slit 1 and slit 2
        *sample_width*      : float | mm
            w, width of the sample
        *sample_broadening* : float | degrees FWHM
            additional divergence caused by sample

    :Returns:
        *dT*  : float OR [float] | degrees FWHM
            calculated angular divergence

    Algorithm:

    Uses the following formula:

        p = w * sin(radians(T))
        dT = /  1/2 (s1+s2) / (d1-d2)   if p >= s2
             \  1/2 (s1+p) /  d1        otherwise
        dT = degrees(dT) + sample_broadening

    where p is the projection of the sample into the beam.

    *sample_broadening* can be estimated from W, the FWHM of a rocking curve:

        sample_broadening = W - degrees( 0.5*(s1+s2) / (d1-d2))

    :Note:
        default sample width is large but not infinite so that at T=0,
        sin(0)*sample_width returns 0 rather than NaN.
    """
    # TODO: check that the formula is correct for T=0 => dT = s1 / d1
    # TODO: add sample_offset and compute full footprint
    d1,d2 = distance
    try:
        s1,s2 = slits
    except TypeError:
        s1=s2 = slits

    # Compute FWHM angular divergence dT from the slits in degrees
    dT = degrees(0.5*(s1+s2)/(d1-d2))

    # For small samples, use the sample projection instead.
    sample_s = sample_width * sin(radians(T))
    if isscalar(sample_s):
        if sample_s < s2: dT = degrees(0.5*(s1+sample_s)/d1)
    else:
        idx = sample_s < s2
        #print s1,s2,d1,d2,T,dT,sample_s
        s1 = ones_like(sample_s)*s1
        dT = ones_like(sample_s)*dT
        dT[idx] = degrees(0.5*(s1[idx] + sample_s[idx])/d1)

    return dT + sample_broadening

def slit_widths(T=None,slits_at_Tlo=None,Tlo=90,Thi=90,
                  slits_below=None, slits_above=None):
    """
    Compute the slit widths for the standard scanning reflectometer
    fixed-opening-fixed geometry.

    :Parameters:
        *T* : [float] | degrees
            Specular measurement angles.
        *Tlo*, *Thi* : float | degrees
            Start and end of the opening region.  The default if *Tlo* is
            not specified is to use fixed slits at *slits_below* for all
            angles.
        *slits_below*, *slits_above* : float OR [float,float] | mm
            Slits outside opening region.  The default is to use the
            values of the slits at the ends of the opening region.
        *slits_at_Tlo : float OR [float,float] | mm
            Slits at the start of the opening region.

    :Returns:
        *s1*, *s2* : [float] | mm
            Slit widths for each theta.

    Slits are assumed to be fixed below angle *Tlo* and above angle *Thi*,
    and opening at a constant dT/T between them.

    Slit openings are defined by a tuple (s1,s2) or constant s=s1=s2.
    With no *Tlo*, the slits are fixed with widths defined by *slits_below*,
    which defaults to *slits_at_Tlo*.  With no *Thi*, slits are continuously
    opening above *Tlo*.

    Note that this function works equally well if angles are measured in
    radians and/or slits are measured in inches.
    """

    # Slits at T<Tlo
    if slits_below is None:
        slits_below = slits_at_Tlo
    try:
        b1,b2 = slits_below
    except TypeError:
        b1=b2 = slits_below
    s1 = ones_like(T) * b1
    s2 = ones_like(T) * b2

    # Slits at Tlo<=T<=Thi
    try:
        m1,m2 = slits_at_Tlo
    except TypeError:
        m1=m2 = slits_at_Tlo
    idx = abs(T) >= Tlo
    s1[idx] = m1 * T[idx]/Tlo
    s2[idx] = m2 * T[idx]/Tlo

    # Slits at T > Thi
    if slits_above is None:
        slits_above = m1 * Thi/Tlo, m2 * Thi/Tlo
    try:
        t1,t2 = slits_above
    except TypeError:
        t1=t2 = slits_above
    idx = abs(T) > Thi
    s1[idx] = t1
    s2[idx] = t2

    return s1,s2


'''
def resolution(Q=None,s=None,d=None,L=None,dLoL=None,Tlo=None,Thi=None,
               s_below=None, s_above=None,
               broadening=0, sample_width=1e10, sample_distance=0):
    """
    Compute the resolution for Q on scanning reflectometers.

    broadening is the sample warp contribution to angular divergence, as
    measured by a rocking curve.  The value should be w - (s1+s2)/(2*d)
    where w is the full-width at half maximum of the rocking curve.

    For itty-bitty samples, provide a sample width w and sample distance ds
    from slit 2 to the sample.  If s_sample = sin(T)*w is smaller than s2
    for some T, then that will be used for the calculation of dT instead.

    """
    T = QL2T(Q=Q,L=L)
    slits = slit_widths(T=T, s=s, Tlo=Tlo, Thi=Thi)
    dT = divergence(T=T,slits=slits, sample_width=sample_width,
                    sample_distance=sample_distance) + broadening
    Q,dQ = Qresolution(L, dLoL*L, T, dT)
    return FWHM2sigma(dQ)

def demo():
    import pylab
    from numpy import linspace, exp, real, conj, sin, radians
    # Values from volfrac example in garefl
    T = linspace(0,9,140)
    Q = 4*pi*sin(radians(T))/5.0042
    dQ = resolution(Q,s=0.21,Tlo=0.35,d=1890.,L=5.0042,dLoL=0.009)
    #pylab.plot(Q,dQ)

    # Fresnel reflectivity for silicon
    rho,sigma=2.07,5
    kz=Q/2
    f = sqrt(kz**2 - 4*pi*rho*1e-6 + 0j)
    r = (kz-f)/(kz+f)*exp(-2*sigma**2*kz*f)
    r[abs(kz)<1e-10] = -1
    R = real(r*conj(r))
    pylab.errorbar(Q,R,xerr=dQ,fmt=',r',capsize=0)
    pylab.grid(True)
    pylab.semilogy(Q,R,',b')

    pylab.show()

def demo2():
    import numpy,pylab
    Q,R,dR = numpy.loadtxt('ga128.refl.mce').T
    dQ = resolution(Q, s=0.154, Tlo=0.36, d=1500., L=4.75, dLoL=0.02)
    pylab.errorbar(Q,R,xerr=dQ,yerr=dR,fmt=',r',capsize=0)
    pylab.grid(True)
    pylab.semilogy(Q,R,',b')
    pylab.show()



if __name__ == "__main__":
    demo2()
'''
