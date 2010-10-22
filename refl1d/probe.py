# This program is in the public domain
# Tuthor: Paul Kienzle

"""
.. sidebar:: On this Page
    
        * :class:`Base probe <refl1d.probe.Probe>`
        * :class:`Polarized neutron probe <refl1d.probe.PolarizedNeutronProbe>`
        * :class:`Neutron probe <refl1d.probe.NeutronProbe>`
        * :class:`X-ray probe <refl1d.probe.XrayProbe>`
   
Experimental probe.

The experimental probe describes the incoming beam for the experiment.
Scattering properties of the sample are dependent on the type and
energy of the radiation.

For time-of-flight measurements, each angle should be represented as
a different probe.  This eliminates the 'stitching' problem, where
Q = 4 pi sin(T1)/L1 = 4 pi sin(T2)/L2 for some (T1,L1) and (T2,L2).
With stitching, it is impossible to account for effects such as
alignment offset since two nominally identical Q values will in
fact be different.  No information is lost treating the two data sets
separately --- each points will contribute to the overall cost function
in accordance with its statistical weight.
"""

# TOF stitching introduces a lot of complexity.
# Theta offset:
#   Q1 = 4 pi sin(T1 + offset)/L1
#   Q2 = 4 pi sin(T2 + offset)/L2
#   at offset=0,
#      Q1=Q2
#   at offset!=0,
#      Q1' ~ Q1 + 4pi offset/L1
#      Q2' ~ Q2 + 4pi offset/L2
#      => Q1' != Q2'
# Thick layers:
#   Since a given Q,dQ has multiple T,dT,L,dL, oversampling is going
#   to be very complicated.
# Energy dependent SLD:
#   Just because two points are at the same Q does not mean they have
#   the same theory function when scattering length density is
#   energy dependent
#
# Not stitching has its own issues.
# Calculation speed:
#   overlapping points are recalculated
#   profiles are recalculated
#
# Unstitched seems like the better bet.

import numpy
from numpy import radians, sin, sqrt, tan, cos, pi, inf, sign, log
from periodictable import nsf, xsf
from .reflectivity import convolve
from . import fresnel
from material import Vacuum
from mystic.parameter import Parameter
from .util import TL2Q, dTdL2dQ

PROBE_KW = ('T', 'dT', 'L', 'dL', 'data',
            'intensity', 'background', 'back_absorption',
            'theta_offset', 'back_reflectivity', 'data')

class Probe(object):
    """
    Defines the incident beam used to study the material.

    The probe is used to compute the scattering potential for the individual
    materials that the beam will bass through.  This potential is normalized
    to density=1 g/cm**3.  To use these values in the calculation of
    reflectivity, they need to be scaled by density and volume fraction.

    The choice of normalized density is dictated by the cost of lookups
    in the CXRO scattering tables.  With normalized density only one lookup
    is necessary during the fit, rather than one for each choice of density
    even when density is not the fitting parameter.

    For calculation purposes, probe needs to return the values, Q_calc, at
    which the model is evaluated.  This is normally going to be the measured
    points only, but for some systems, such as those with very thick layers,
    oversampling is needed to avoid aliasing effects.

    Measurement properties:

        *intensity* is the beam intensity
        *background* is the background
        *back_absorption* is the amount of absorption through the substrate
        *theta_offset* is the offset of the sample from perfect alignment

    Measurement properties are fittable parameters.  *theta_offset* in
    particular should be set using probe.theta_offset.dev(dT), with dT
    equal to the uncertainty in the peak position for the rocking curve,
    as measured in radians.  Changes to *theta_offset* will then be penalized
    in the cost function for the fit as if it were another measurement.  Note
    that the uncertainty in the peak position is not the same as the width
    of the peak.  The peak stays roughly the same as statistics are improved,
    but the uncertainty in position and width will decrease. [#Daymond2002]_
    There is an additional uncertainty in the angle due to motor step size,
    easily computed from the variance in a uniform distribution.  Combined,
    the uncertainty in *theta_offset* is::

        >>> dT = w/sqrt(I) + d/sqrt(12)

    where *w* is the full-width of the peak in radians at half maximum, *I*
    is the integrated intensity under the peak and *d* is the motor step size
    is radians.

    *intensity* and *back_absorption* are generally not needed --- scaling
    the reflected signal by an appropriate intensity measurement will correct
    for both of these during reduction.  *background* may be needed,
    particularly for samples with significant hydrogen content due to its
    large isotropic incoherent scattering cross section.

    View properties:

        *substrate* is the material which makes up the substrate
        *surface* is the material which makes up the surface
        *view* is 'fresnel', 'log', 'linear' or 'q4'

    Normally *view* is set directly in the class rather than the
    instance since it is not specific to the view.  The fresnel
    substrate and surface materials are a property of the sample,
    and should share the same material.

    .. [#Daymond2002] M.R. Daymond, P.J. Withers and M.W. Johnson;
       The expected uncertainty of diffraction-peak location",
       Appl. Phys. A 74 [Suppl.], S112 - S114 (2002).
       http://dx.doi.org/10.1007/s003390201392
    """
    polarized = False
    view = "fresnel"
    substrate = None
    surface = None
    def __init__(self, T=None, dT=0, L=None, dL=0, data = None,
                 intensity=1, background=0, back_absorption=1, theta_offset=0,
                 back_reflectivity=False):
        if T is None or L is None:
            raise TypeError("T and L required")
        self.intensity = Parameter.default(intensity,name="intensity")
        self.background = Parameter.default(background,name="background",
                                            limits=[0,inf])
        self.back_absorption = Parameter.default(back_absorption,
                                                 name="back_absorption",
                                                 limits=[0,1])
        self.theta_offset = Parameter.default(theta_offset,name="theta_offset")

        self.back_reflectivity = back_reflectivity

        #if L is None:
        #    L = xsf.xray_wavelength(E)
        #    dL = L * dE/E
        #else:
        #    E = xsf.xray_energy(L)
        #    dE = E * dL/L

        Q = TL2Q(T=T, L=L)
        dQ = dTdL2dQ(T=T,dT=dT,L=L,dL=dL)

        # Make sure that we are dealing with vectors
        T,dT,L,dL = [numpy.ones_like(Q)*v for v in (T,dT,L,dL)]

        # Probe stores sorted values for convenience of resolution calculator
        idx = numpy.argsort(Q)
        self.T, self.dT = T[idx],dT[idx]
        self.L, self.dL = L[idx],dL[idx]
        self.Qo, self.dQ = Q[idx],dQ[idx]
        if data is not None:
            R,dR = data
            if R is not None: R = R[idx]
            if dR is not None: dR = dR[idx]
            self.Ro = self.R = R
            self.dR = dR

        # By default the calculated points are the measured points.  Use
        # oversample() for a more accurate resolution calculations.
        self._set_calc(self.T,self.L)
        self.data = data

    def log10_to_linear(self):
        """
        Convert data from log to linear.

        Older reflectometry reduction code stored reflectivity in log base 10
        format.  Call probe.log10_to_linear() after loading this data to
        convert it to linear for subsequent display and fitting.
        """
        if self.Ro != None:
            self.Ro = 10**self.Ro
            if self.dR != None:
                self.dR = self.Ro * self.dR * log(10)
            self.R = self.Ro

    def _get_data(self):
        return self.R0,self.dR
    def _set_data(self, data):
        if data != None:
            self.R,self.dR = data
            # Remember the original so we can resynthesize as needed
            self.Ro = self.R
    data = property(_get_data, _set_data)
    def resynth_data(self):
        """
        Generate new data according to the model R ~ N(Ro,dR).

        The resynthesis step is a precursor to refitting the data, as is
        required for certain types of monte carlo error analysis.
        """
        self.R = self.Ro + numpy.random.randn(*self.Ro.shape)*self.dR

    def restore_data(self):
        """
        Restore the original data.
        """
        self.R = self.Ro

    def _set_calc(self, T, L):
        Q = TL2Q(T=T, L=L)

        idx = numpy.argsort(Q)
        self.calc_T = T[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self._sf_L = numpy.unique(self.calc_L)
        self._sf_idx = numpy.searchsorted(self._sf_L, L)

    def _Q(self):
        if self.theta_offset.value != 0:
            Q = TL2Q(T=self.T+self.theta_offset.value, L=self.L)
        else:
            Q = self.Qo
        return Q
    Q = property(_Q)
    def _calc_Q(self):
        if self.theta_offset.value != 0:
            Q = TL2Q(T=self.calc_T+self.theta_offset.value, L=self.calc_L)
        else:
            Q = self.calc_Qo
        return Q if not self.back_reflectivity else -Q
    calc_Q = property(_calc_Q)
    def parameters(self):
        return dict(intensity=self.intensity,
                    background=self.background,
                    backabsorption=self.back_absorption,
                    theta_offset=self.theta_offset)
    def scattering_factors(self, material):
        """
        Returns the scattering factors associated with the material given
        the range of wavelengths/energies used in the probe.
        """
        raise NotImplementedError
    def __len__(self):
        """
        Returns the number of scattering factors that will be returned for
        the probe.
        """
        return len(self._sf_L)
    def oversample(self, n=6, seed=1):
        """
        Generate an over-sampling of Q to avoid aliasing effects.

        Oversampling is needed for thick layers, in which the underlying
        reflectivity oscillates so rapidly in Q that a single measurement
        has contributions from multiple Kissig fringes.

        Sampling will be done using a pseudo-random generator so that
        accidental structure in the function does not contribute to the
        aliasing.  The generator will usually be initialized with a fixed
        *seed* so that the point selection will not change from run to run,
        but a *seed* of None will choose a different set of points each time
        oversample is called.

        The value *n* is the number of points that should contribute to
        each Q value when computing the resolution.   These will be
        distributed about the nominal measurement value, but varying in
        both angle and energy according to the resolution function.  This
        will yield more points near the measurement and fewer farther away.
        The measurement point itself will not be used to avoid accidental
        bias from uniform Q steps.
        """
        rng = numpy.random.RandomState(seed=seed)
        T = rng.normal(self.T[:,None],self.dT[:,None],size=(len(self.dT),n))
        L = rng.normal(self.L[:,None],self.dL[:,None],size=(len(self.dL),n))
        T = T.flatten()
        L = L.flatten()
        self._set_calc(T,L)

    def apply_beam(self, calc_Q, calc_R, resolution=True):
        """
        Apply factors such as beam intensity, background, backabsorption,
        resolution and footprint to the data.

        Users who wish to create complex resolution functions, e.g.,
        with wavelength following the TOF feather will need to control
        both the sampling and the resolution calculation.  Probe is the
        natural place for this calculation since it controls both of these.
        """
        # Handle absorption through the substrate, which occurs when Q<0
        # (condition)*C is C when condition is True or 0 when False,
        # (condition)*(C-1)+1 is C when condition is True or 1 when False.
        back = (calc_Q<0)*(self.back_absorption.value-1)+1
        calc_R *= back

        # For back reflectivity, reverse the sign of Q after computing
        if self.back_reflectivity:
            calc_Q = -calc_Q
        if resolution:
            if calc_Q[-1] < calc_Q[0]:
                calc_Q, calc_R = [v[::-1] for v in calc_Q, calc_R]
            Q,R = self.Q, convolve(calc_Q, calc_R, self.Q, self.dQ)
        else:
            Q,R = calc_Q, calc_R
        R = self.intensity.value*R + self.background.value
        return Q,R

    def fresnel(self, substrate=None, surface=None):
        """
        Compute the reflectivity for the probe reflecting from a block of
        material with the given substrate.

        Returns F = R(probe.Q), where R is magnitude squared reflectivity.
        """
        # Doesn't use ProbeCache, but this routine is not time critical
        Srho,Sirho = (0,0) if substrate is None else substrate.sld(self)[:2]
        Vrho,Virho = (0,0) if surface is None else surface.sld(self)[:2]
        I = numpy.ones_like(self.Q)
        calculator = fresnel.Fresnel(rho=Srho*I, irho=Sirho*I,
                                     Vrho=Vrho*I, Virho=Virho*I)
        return calculator(Q=self.Q)

    def plot(self, theory=None, substrate=None, surface=None, view=None):
        """
        Plot theory against data.

        Need substrate/surface for Fresnel reflectivity
        """
        view = view if view is not None else self.view
        if view == 'linear':
            self.plot_linear(theory=theory)
        elif view == 'log':
            self.plot_log(theory=theory)
        elif view == 'fresnel':
            self.plot_fresnel(theory=theory, substrate=substrate,
                              surface=surface)
        elif view == 'q4':
            self.plot_Q4(theory=theory)
        elif view == 'resolution':
            self.plot_resolution()
        else:
            raise TypeError("incorrect reflectivity view '%s'"%self.view)

    def plot_resolution(self, theory=None):
        import pylab
        pylab.plot(self.Q, self.dQ)
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('Q resolution (1-sigma inv Angstroms)')
        pylab.title('Measurement resolution')


    def plot_linear(self, theory=None):
        """
        Plot the data associated with probe.
        """
        import pylab
        if hasattr(self, 'R') and self.R is not None:
            pylab.errorbar(self.Q, self.R,
                           yerr=self.dR, xerr=self.dQ, fmt='.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R, hold=True)
        pylab.yscale('linear')
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('Reflectivity')
    def plot_log(self, theory=None):
        """
        Plot the data associated with probe.
        """
        import pylab
        if hasattr(self,'R') and self.R is not None:
            pylab.errorbar(self.Q, self.R,
                           yerr=self.dR, xerr=self.dQ, fmt='.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R, hold=True)
        pylab.yscale('log')
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('Reflectivity')
    def plot_fresnel(self, theory=None, substrate=None, surface=None):
        """
        Plot the Fresnel reflectivity associated with the probe.
        """
        import pylab
        if substrate is None and surface is None:
            raise TypeError("Fresnel reflectivity needs substrate or surface")
        F = self.fresnel(substrate=substrate,surface=surface)
        F *= self.intensity.value
        if hasattr(self,'R') and self.R is not None:
            pylab.errorbar(self.Q, self.R/F, self.dR/F, self.dQ, '.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R/F, hold=True)
        pylab.xlabel('Q (inv Angstroms)')
        if substrate is None:
            name = "air:%s"%(surface.name)
        elif surface is None or isinstance(surface,Vacuum):
            name = substrate.name
        else:
            name = "%s:%s"%(substrate.name, surface.name)
        pylab.ylabel('R/R(%s)'%(name))
    def plot_Q4(self, theory=None):
        """
        Plot the Q**4 reflectivity associated with the probe.
        """
        import pylab
        Q4 = 1e8*self.Q**4
        Q4 /= self.intensity.value
        #Q4[Q4==0] = 1
        if hasattr(self,'R') and self.R is not None:
            pylab.errorbar(self.Q, self.R*Q4, self.dR*Q4, self.dQ, '.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R*Q4, hold=True)
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('R (100 Q)^4')

class XrayProbe(Probe):
    """
    X-Ray probe.

    Contains information about the kind of probe used to investigate
    the sample.

    X-ray data is traditionally recorded by angle and energy, rather
    than angle and wavelength as is used by neutron probes.
    """
    def scattering_factors(self, material):
        # doc string is inherited from parent (see below)
        rho, irho = xsf.xray_sld(material,
                                 wavelength = self._sf_L,
                                 density=1)
        # TODO: support wavelength dependent systems
        return rho[0], irho[0], 0
        return rho[self._sf_idx], irho[self._sf_idx], 0
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

class NeutronProbe(Probe):
    def scattering_factors(self, material):
        # doc string is inherited from parent (see below)
        rho, irho, rho_incoh = nsf.neutron_sld(material,
                                               wavelength=self._sf_L,
                                               density=1)
        # TODO: support wavelength dependent systems
        return rho, irho[0], rho_incoh
        return rho, irho[self._sf_idx], rho_incoh
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__


def measurement_union(xs):
    TL = set()
    for x in xs:
        if x is not None:
            TL = TL | set(zip(x.T,x.dT,x.L,x.dL))
    T,dT,L,dL = [numpy.array(sorted(v))
                 for v in zip(*[w for w in TL])]
    Q = TL2Q(T,L)
    dQ = dTdL2dQ(T,dT,L,dL)
    idx = numpy.argsort(Q)
    return T[idx],dT[idx],L[idx],dL[idx],Q[idx],dQ[idx]

class PolarizedNeutronProbe(object):
    """
    Polarized neutron probe

    *xs* (4 x NeutronProbe) is a sequence pp, pm, mp and mm.
    *Tguide* (degrees) is the angle of the guide field
    """
    view = None  # Default to Probe.view so only need to change in one place
    substrate = surface = None
    polarized = True
    def __init__(self, xs=None, Tguide=270):
        self.pp, self.pm, self.mp, self.mm = xs

        self.T, self.dT, self.L, self.dL, self.Q, self.dQ \
            = measurement_union(xs)
        self._set_calc(self.T, self.L)

    def shared_beam(self, intensity=1, background=0,
                    back_absorption=1, theta_offset=0):
        """
        Share beam parameters across all four cross sections.

        New parameters are created for *intensity*, *background*,
        *theta_offset* and *back_absorption* and assigned to the all
        cross sections.  These can be replaced in an individual
        cross section if for some reason one of the parameters is
        independent.
        """
        intensity = Parameter.default(intensity,name="intensity")
        background = Parameter.default(background,name="background",
                                       limits=[0,inf])
        back_absorption = Parameter.default(back_absorption,
                                            name="back_absorption",
                                            limits=[0,1])
        theta_offset = Parameter.default(theta_offset,name="theta_offset")
        for x in self.pp, self.pm, self.mp, self.mm:
            if x is not None:
                x.intensity = intensity
                x.background = background
                x.back_absorption = back_absorption
                x.theta_offset = theta_offset

    def oversample(self, n=6, seed=1):
        # doc string is inherited from parent (see below)
        rng = numpy.random.RandomState(seed=seed)
        T = rng.normal(self.T[:,None],self.dT[:,None],size=(len(self.dT),n))
        L = rng.normal(self.L[:,None],self.dL[:,None],size=(len(self.dL),n))
        T = T.flatten()
        L = L.flatten()
        self._set_calc(T,L)
    oversample.__doc__ = Probe.oversample.__doc__

    def _calc_Q(self):
        for x in self.pp, self.pm, self.mp, self.mm:
            if x is not None:
                return x.calc_Q
        raise RuntimeError("No polarization cross sections")
    calc_Q = property(_calc_Q)

    def _set_calc(self, T, L):
        """
        Propagate setting of calc_Q to the individual cross sections.
        """
        for x in self.pp, self.pm, self.mp, self.mm:
            if x is not None:
                x._set_calc(T,L)

    def apply_beam(self, R, resolution=True):
        """
        Apply factors such as beam intensity, background, backabsorption,
        and footprint to the data.
        """
        for xs in self.pp, self.pm, self.mp, self.mm:
            if xs is not None:
                xs.apply_beam(R, resolution)

    def scattering_factors(self, material):
        # doc string is inherited from parent (see below)
        rho, irho, rho_incoh = nsf.neutron_sld(material,
                                               wavelength=self._sf_L,
                                               density=1)
        return rho, irho[self._sf_idx], rho_incoh
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def __len__(self):
        return len(self.calc_Q)

    def plot(self, theory=None, substrate=None, surface=None, view=None):
        """
        Plot theory against data.

        Need substrate/surface for Fresnel reflectivity
        """
        view = view if view is not None else self.view
        if view is None: view = Probe.view  # Default to Probe.view

        if view == 'linear':
            self.plot_linear(theory=theory)
        elif view == 'log':
            self.plot_log(theory=theory)
        elif view == 'fresnel':
            self.plot_fresnel(theory=theory,
                              substrate=substrate, surface=surface)
        elif view == 'q4':
            self.plot_Q4(theory=theory)
        elif view == 'SA':
            self.plot_SA(theory=theory)
        elif view == 'resolution':
            self.plot_resolution()
        else:
            raise TypeError("incorrect reflectivity view '%s'"%self.view)

    def plot_resolution(self, theory=None):
        self._xs_plot('plot_resolution', theory=theory)
    def plot_linear(self, theory=None):
        self._xs_plot('plot_linear', theory=theory)
    def plot_log(self, theory=None):
        self._xs_plot('plot_log', theory=theory)
    def plot_fresnel(self, theory=None, substrate=None, surface=None):
        self._xs_plot('plot_fresnel', theory=theory,
                      substrate=substrate, surface=surface)
    def plot_Q4(self, theory=None):
        self._xs_plot('plot_Q4', theory=theory)
    def plot_SA(self, theory):
        import pylab
        if self.pp is None or self.mm is None:
            raise TypeError("cannot form spin asymmetry plot with ++ and --")

        pp,mm = self.pp,self.mm
        if hasattr(pp,'R'):
            Q,SA,dSA = spin_asymmetry(pp.Q,pp.R,pp.dR,mm.Q,mm.R,mm.dR)
            if dSA is not None:
                pylab.errorbar(Q, SA, yerr=dSA, xerr=pp.dQ, fmt='.')
            else:
                pylab.plot(Q,SA,'.')
        if theory is not None:
            Q,pp,pm,mp,mm = theory
            Q,SA,_ = spin_asymmetry(Q,pp,None,Q,mm,None)
            pylab.plot(Q, SA, hold=True)
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel(r'spin asymmetry $(R^+ -\, R^-) / (R^+ +\, R^-)$')

    def _xs_plot(self, plotter, theory=None, **kw):
        import pylab
        # Plot available cross sections
        isheld = pylab.ishold()
        if theory is not None:
            Q,pp,pm,mp,mm = theory
            theory = ((Q,pp),(Q,pm),(Q,mp),(Q,pm))
        else:
            theory = (None,None,None,None)
        for xs,xstheory in zip((self.pp, self.pm, self.mp, self.mm),theory):
            if xs is not None:
                fn = getattr(xs, plotter)
                fn(theory=xstheory, **kw)
                pylab.hold(True)
        if not isheld: pylab.hold(False)

def spin_asymmetry(Qp,Rp,dRp,Qm,Rm,dRm):
    """
    Compute spin asymmetry for R+,R-.

    Returns *Q*, *SA*, *dSA*.

    Spin asymmetry, *SA*, is::

        >>> SA = (Rp - Rm)/(Rp + Rm)

    Uncertainty *dSA* follows from propagation of error::

        >>> dSA^2 = 4(Rp^2  dRm^2  -  Rm^2 dRp^2)/(Rp + Rm)^4

    The inputs (*Qp*, *Rp*, *dRp*) and (*Qm*, *Rm*, *dRm*) are measurements
    for the ++ and -- cross sections respectively.  If *dRp*, *dRm* are None,
    then the returned uncertainty will also be None.
    """
    Rm = numpy.interp(Qp,Qm,Rm)
    v = (Rp-Rm)/(Rp+Rm)
    if dRp is not None:
        dRm = numpy.interp(Qp,Qm,dRm)
        dvsq = 4 * ((Rp*dRm)**2 - (Rm*dRp)**2) / (Rp+Rm)**4
        return Qp, v, sqrt(dvsq)
    else:
        return Qp, v, None




class ProbeSet(Probe):
    def __init__(self, probes):
        self.probes = probes
        self.R = numpy.hstack(p.R for p in self.probes)
        self.dR = numpy.hstack(p.dR for p in self.probes)
        self._len = sum([len(p) for p in self.probes])

    def parameters(self):
        return [p.paramters() for p in self.probes]
    parameters.__doc__ = Probe.parameters.__doc__

    def resynth_data(self):
        for p in self.probes: p.resynth_data()
        self.R = numpy.hstack(p.R for p in self.probes)
    resynth_data.__doc__ = Probe.resynth_data.__doc__

    def restore_data(self):
        for p in self.probes: p.restore_data()
        self.R = numpy.hstack(p.R for p in self.probes)
    restore_data.__doc__ = Probe.restore_data.__doc__

    def __len__(self):
        return self._len
    def _Q(self):
        return numpy.hstack(p.Q for p in self.probes)
    Q = property(_Q)
    def _calc_Q(self):
        return numpy.unique(numpy.hstack(p.calc_Q for p in self.probes))
    calc_Q = property(_calc_Q)
    def oversample(self, **kw):
        for p in self.probes: p.oversample(**kw)
    oversample.__doc__ = Probe.oversample.__doc__
    def scattering_factors(self, material):
        return self.probes[0].scattering_factors(material)
        result = [p.scattering_factors(material) for p in self.probes]
        return [numpy.hstack(v) for v in zip(*result)]
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__
    def apply_beam(self, calc_Q, calc_R, **kw):
        result = [p.apply_beam(calc_Q, calc_R, **kw) for p in self.probes]
        return [numpy.hstack(v) for v in zip(*result)]
    def plot(self, theory=None, **kw):
        for p,th in self._plotparts(theory): p.plot(theory=th, **kw)
    plot.__doc__ = Probe.plot.__doc__
    def plot_resolution(self, **kw):
        for p in self.probes: p.plot_resolution(**kw)
    plot_resolution.__doc__ = Probe.plot_resolution.__doc__
    def plot_linear(self, theory=None, **kw):
        for p,th in self._plotparts(theory): p.plot_linear(theory=th, **kw)
    plot_linear.__doc__ = Probe.plot_linear.__doc__
    def plot_log(self, theory=None, **kw):
        for p,th in self._plotparts(theory): p.plot_log(theory=th, **kw)
    plot_log.__doc__ = Probe.plot_log.__doc__
    def plot_fresnel(self, theory=None, **kw):
        for p,th in self._plotparts(theory): p.plot_fresnel(theory=th, **kw)
    plot_fresnel.__doc__ = Probe.plot_fresnel.__doc__
    def plot_Q4(self, theory=None, **kw):
        for p,th in self._plotparts(theory): p.plot_Q4(theory=th, **kw)
    plot_Q4.__doc__ = Probe.plot_Q4.__doc__
    def _plotparts(self, theory):
        if theory == None:
            for p in self.probes: yield p
        else:
            offset = 0
            Q,R = theory
            for p in self.probes:
                n = len(p)
                yield p,(Q[offset:offset+n],R[offset:offset+n])
                offset += n

    def shared_beam(self, intensity=1, background=0,
                    back_absorption=1, theta_offset=0):
        """
        Share beam parameters across all segments.

        New parameters are created for *intensity*, *background*,
        *theta_offset* and *back_absorption* and assigned to the all
        segments.  These can be replaced in an individual segment if
        that parameter is independent.
        """
        intensity = Parameter.default(intensity,name="intensity")
        background = Parameter.default(background,name="background",
                                       limits=[0,inf])
        back_absorption = Parameter.default(back_absorption,
                                            name="back_absorption",
                                            limits=[0,1])
        theta_offset = Parameter.default(theta_offset,name="theta_offset")
        for p in self.probes:
            p.intensity = intensity
            p.background = background
            p.back_absorption = back_absorption
            p.theta_offset = theta_offset
    def stitch(self, tol=0.01):
        r"""
        Stitch together multiple datasets into a single dataset.

        *Not implemented*

        Points within *tol* of each other and with the same resolution
        are combined by interpolating them to a common Q value then averaged
        using Gaussian error propagation.

        :Returns: probe | Probe
            Combined data set.

        :Algorithm:

        To interpolate a set of points to a common value, first find the
        common *Q* value:

        .. math::

            \hat Q = \sum{Q_k} / n

        Then for each dataset *k*, find the interval [i,i+1] containing the
        value *Q*, and use it to compute interpolated value for *R*:

        .. math::

            w_k &=& (\hat Q - Q_k_i)/(Q_k_{i+1} - Q_k_i) \\
            \hat R_k &=& w_k R_k_{i+1} + (1-w_k) R_k_{i+1} \\
            \hat \sigma_{R_k} &=& \sqrt{ w^2_k \sigma^2_{R_k_i} + (1-w_k)^2 \sigma^2_{R_k_{i+1}} } / n

        Average the resulting *R* using Gaussian error propagation:

        .. math::

            \hat R &=& \sum{\hat R_k}/n \\
            \hat \sigma_R &=& \sqrt{\sum \hat \sigma_{R_k}^2}/n

        """
        raise NotImplementedError
