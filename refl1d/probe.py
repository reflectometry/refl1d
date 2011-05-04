# This program is in the public domain
# Author: Paul Kienzle
r"""
Experimental probe.

The experimental probe describes the incoming beam for the experiment.
Scattering properties of the sample are dependent on the type and
energy of the radiation.

See `data-guide`_ for details.

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

from __future__ import with_statement, division
import os
import numpy
from numpy import radians, sin, sqrt, tan, cos, pi, inf, sign, log
from periodictable import nsf, xsf
from .reflectivity import convolve
from . import fresnel
from material import Vacuum
from mystic.parameter import Parameter, Constant
from .resolution import TL2Q, dTdL2dQ
from .stitch import stitch

PROBE_KW = ('T', 'dT', 'L', 'dL', 'data',
            'intensity', 'background', 'back_absorption',
            'theta_offset', 'back_reflectivity', 'data')

def make_probe(**kw):
    """
    Return a reflectometry measurement object of the given resolution.
    """
    radiation = kw.pop('radiation')
    kw = dict((k,v) for k,v in kw.items() if k in PROBE_KW)
    if radiation == 'neutron':
        return NeutronProbe(**kw)
    else:
        return XrayProbe(**kw)


class Probe(object):
    r"""
    Defines the incident beam used to study the material.

    For calculation purposes, probe needs to return the values $Q_\text{calc}$
    at which the model is evaluated.  This is normally going to be the measured
    points only, but for some systems, such as those with very thick layers,
    oversampling is needed to avoid aliasing effects.

    Measurement properties:

        *intensity* is the beam intensity
        *background* is the background
        *back_absorption* is the amount of absorption through the substrate
        *theta_offset* is the offset of the sample from perfect alignment
        *back_reflectivity* is true if the beam enters through the substrate

    Measurement properties are fittable parameters.  *theta_offset* in
    particular should be set using probe.theta_offset.dev(dT), with dT
    equal to the uncertainty in the peak position for the rocking curve,
    as measured in radians.  Changes to *theta_offset* will then be penalized
    in the cost function for the fit as if it were another measurement.  Use
    :meth:`alignment_uncertainty` to compute dT from the shape of the
    rocking curve.

    *intensity* and *back_absorption* are generally not needed --- scaling
    the reflected signal by an appropriate intensity measurement will correct
    for both of these during reduction.  *background* may be needed,
    particularly for samples with significant hydrogen content due to its
    large isotropic incoherent scattering cross section.

    View properties:

        *substrate* is the material which makes up the substrate
        *surface* is the material which makes up the surface
        *view* is 'fresnel', 'log', 'linear', 'q4', 'residual'

    Normally *view* is set directly in the class rather than the
    instance since it is not specific to the view.  The fresnel
    substrate and surface materials are a property of the sample,
    and should share the same material.
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
        if data is not None:
            R,dR = data
        else:
            R,dR = None,None

        self._set_TLR(T,dT,L,dL,R,dR)

    def _set_TLR(self, T,dT,L,dL,R,dR):
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
        if R is not None: R = R[idx]
        if dR is not None: dR = dR[idx]
        self.Ro = self.R = R
        self.dR = dR

        # By default the calculated points are the measured points.  Use
        # oversample() for a more accurate resolution calculations.
        self._set_calc(self.T,self.L)

    @staticmethod
    def alignment_uncertainty(w,I,d=0):
        r"""
        Compute alignment uncertainty.

        **Parameters:**

        *w* : float | degrees
            Rocking curve full width at half max.
        *I* : float | counts
            Rocking curve integrated intensity.
        *d* = 0: float | degrees
            Motor step size

        **Returns:**

        *dtheta* : float | degrees
            uncertainty in alignment angle
        """
        return sqrt(w**2/I + d**2/12.)


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
        raise NotImplementedError
        return self.Ro,self.dR
    def _set_data(self, data):
        # Setting data is dangerous since the Q points may have been
        # reordered to keep them sorted.  The external user may be
        # expecting to reset them from the original data file, which
        # would fail, or from a simulation based on Q, which would
        # succeed.  Storing the reordering vector wouldn't help since
        # it would just reverse the problem and still lead to confusion.
        # Returning self.Q in the original order would allow both cases
        # to work, but it makes simulation more complicated because the
        # resolution calculation wants sorted inputs.  We could just
        # make the resolution calculation sort its inputs, but that
        # increases the cost of computing the reflectivity curve (though
        # that cost may small compared to the cost of sorting an almost
        # sorted list).
        raise NotImplementedError
        if data is not None:
            self.R,self.dR = data
        else:
            self.R = self.dR = None
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

    def simulate_data(self, R, dR):
        """
        Set the data for the probe to R, adding random noise dR.
        """
        self.Ro, self.dR = R+0, dR+0
        self.Ro[R==0] = 1e-10
        self.dR[dR==0] = 1e-11
        self.resynth_data()
        self.Ro = self.R

    def restore_data(self):
        """
        Restore the original data.
        """
        self.R = self.Ro

    def write_data(self, filename,
                   columns=['Q','R','dR'],
                   header=None):
        """
        Save the data to a file.

        *header* is a string with trailing \\n containing the file header.
        *columns* is a list of column names from Q, dQ, R, dR, L, dL, T, dT.

        The default is to write Q,R,dR data.
        """
        if header == None:
            header = "# %s\n"%' '.join(columns)
        with open(filename,'w') as fid:
            fid.write(header)
            data = numpy.vstack([getattr(self,c) for c in columns])
            numpy.savetxt(fid,data.T)

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
            #TODO: this may break the Q order on measurements with varying L
        else:
            Q = self.Qo
        return Q
    Q = property(_Q)
    def _calc_Q(self):
        if self.theta_offset.value != 0:
            Q = TL2Q(T=self.calc_T+self.theta_offset.value, L=self.calc_L)
            #TODO: this may break the Q order on measurements with varying L
        else:
            Q = self.calc_Qo
        return Q if not self.back_reflectivity else -Q
    calc_Q = property(_calc_Q)
    def parameters(self):
        return dict(intensity=self.intensity,
                    background=self.background,
                    back_absorption=self.back_absorption,
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

    def subsample(self, dQ):
        """
        Select points at most every dQ.

        Use this to speed up computation early in the fitting process.

        This changes the data object, and is not reversible.

        The current algorithm is not picking the "best" Q value, just the
        nearest, so if you have nearby Q points with different quality
        statistics (as happens in overlapped regions from spallation
        source measurements at different angles), then it may choose
        badly.  Simple solutions based on the smallest relative error dR/R
        will be biased toward peaks, and smallest absolute error dR will
        be biased toward valleys.
        """
        # Assumes self contains sorted Qo and associated T,L
        # Note: calc_Qo is already sorted
        Q = numpy.arange(self.Qo[0],self.Qo[-1],dQ)
        idx = numpy.unique(numpy.searchsorted(self.Qo,Q))
        #print len(idx),len(self.Qo)

        self.T, self.dT = self.T[idx],self.dT[idx]
        self.L, self.dL = self.L[idx],self.dL[idx]
        self.Qo, self.dQ = self.Qo[idx],self.dQ[idx]
        if self.R is not None: self.Ro = self.R = self.R[idx]
        if self.dR is not None: self.dR = self.dR[idx]
        self._set_calc(self.T,self.L)

    def resolution_guard(self):
        r"""
        Make sure each measured $Q$ point has at least 5 calculated $Q$
        points contributing to it in the range $[-3\Delta Q,3\Delta Q]$.
        """
        raise NotImplementedError
        # TODO: implement resolution guard.

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

    def _apply_resolution(self, Qin, Rin):
        """
        Apply the instrument resolution function
        """
        return self.Q, convolve(Qin, Rin, self.Q, self.dQ)

    def apply_beam(self, calc_Q, calc_R, resolution=True):
        """
        Apply factors such as beam intensity, background, backabsorption,
        resolution to the data.
        """
        # Note: in-place vector operations are not notably faster.

        # Handle absorption through the substrate, which occurs when Q<0
        # (condition)*C is C when condition is True or 0 when False,
        # (condition)*(C-1)+1 is C when condition is True or 1 when False.
        back = (calc_Q<0)*(self.back_absorption.value-1)+1
        calc_R = calc_R * back

        # For back reflectivity, reverse the sign of Q after computing
        if self.back_reflectivity:
            calc_Q = -calc_Q
        if resolution:
            if calc_Q[-1] < calc_Q[0]:
                calc_Q, calc_R = [v[::-1] for v in calc_Q, calc_R]
            Q,R = self._apply_resolution(calc_Q, calc_R)
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
        if self.back_reflectivity:
            Srho,Vrho = Vrho,Srho
            Sirho,Virho = Virho,Sirho
        if Srho == Vrho: Srho = Vrho + 1
        I = numpy.ones_like(self.Q)
        calculator = fresnel.Fresnel(rho=Srho*I, irho=Sirho*I,
                                     Vrho=Vrho*I, Virho=Virho*I)
        return calculator(Q=self.Q)

    def plot(self, view=None, **kwargs):
        """
        Plot theory against data.

        Need substrate/surface for Fresnel reflectivity
        """

        view = view if view is not None else self.view

        if view == 'linear':
            self.plot_linear(**kwargs)
        elif view == 'log':
            self.plot_log(**kwargs)
        elif view == 'fresnel':
            self.plot_fresnel(**kwargs)
        elif view == 'q4':
            self.plot_Q4(**kwargs)
        elif view == 'resolution':
            self.plot_resolution(**kwargs)
        elif view == 'residual':
            self.plot_residuals(**kwargs)
        else:
            raise TypeError("incorrect reflectivity view '%s'"%view)


    def plot_resolution(self, **kwargs):
        import pylab
        pylab.plot(self.Q, self.dQ, label=self.name())
        pylab.xlabel(r'Q ($\AA^{-1}$)')
        pylab.ylabel(r'Q resolution ($1-\sigma \AA^{-1}$)')
        pylab.title('Measurement resolution')


    def plot_linear(self, **kwargs):
        """
        Plot the data associated with probe.
        """
        import pylab
        self._plot_pair(scale=1, ylabel='Reflectivity', **kwargs)
        pylab.yscale('linear')

    def plot_log(self, **kwargs):
        """
        Plot the data associated with probe.
        """
        import pylab
        self._plot_pair(scale=1, ylabel='Reflectivity', **kwargs)
        pylab.yscale('log')

    def plot_fresnel(self, substrate=None, surface=None, **kwargs):
        """
        Plot the Fresnel reflectivity associated with the probe.
        """
        import pylab
        if substrate is None and surface is None:
            raise TypeError("Fresnel reflectivity needs substrate or surface")
        F = self.fresnel(substrate=substrate,surface=surface)
        F *= self.intensity.value
        if substrate is None:
            name = "air:%s"%(surface.name)
        elif surface is None or isinstance(surface,Vacuum):
            name = substrate.name
        else:
            name = "%s:%s"%(substrate.name, surface.name)
        self._plot_pair(scale=1/F, ylabel='R/R(%s)'%(name), **kwargs)

    def plot_Q4(self, **kwargs):
        """
        Plot the Q**4 reflectivity associated with the probe.
        """
        import pylab
        Q4 = 1e8*self.Q**4
        Q4 /= self.intensity.value
        #Q4[Q4==0] = 1
        self._plot_pair(scale=Q4, ylabel='R (100 Q)^4', **kwargs)

    def plot_residuals(self, theory=None, suffix='', **kwargs):
        import matplotlib.pyplot as plt
        if theory is not None and self.R is not None:
            from .util import coordinated_colors
            c = coordinated_colors()
            Q,R = theory
            residual = (R - self.R)/self.dR
            plt.plot(self.Q, residual,
                     '.', color=c['light'],
                     label=self.name()+suffix)
        plt.axhline(1, color='black', ls='--',lw=1)
        plt.axhline(0, color='black', lw=1)
        plt.axhline(-1, color='black', ls='--',lw=1)
        plt.xlabel('Q (inv A)')
        plt.ylabel('(theory-data)/error')

    def _plot_pair(self, theory=None, scale=1, ylabel="", suffix="", **kw):
        import pylab
        from .util import coordinated_colors
        c = coordinated_colors()
        isheld = pylab.ishold()
        if hasattr(self,'R') and self.R is not None:
            pylab.errorbar(self.Q, self.R*scale,
                           yerr=self.dR*scale, xerr=self.dQ,
                           fmt='.', color=c['light'],
                           label=self.name('data')+suffix)
            pylab.hold(True)
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R*scale, color=c['dark'],
                       label=self.name('theory')+suffix)
        pylab.hold(isheld)
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel(ylabel)
        pylab.legend()

    def name(self, gloss="", suffix=""):
        if hasattr(self, 'filename'):
            prefix = os.path.splitext(os.path.basename(self.filename))[0]
            return " ".join((prefix,gloss)) if gloss else prefix
        else:
            return gloss if gloss else None


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

def Qmeasurement_union(xs):
    Qset = set()
    for x in xs:
        if x is not None:
            Qset = Qset | set(zip(x.Q,x.dQ))
    Q,dQ = [numpy.array(sorted(v))
            for v in zip(*[w for w in Qset])]
    idx = numpy.argsort(Q)
    return Q[idx],dQ[idx]


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
        self._check()

    def _check(self):
        back_refls = [f.back_reflectivity
                      for f in (self.pp, self.pm, self.mp, self.mm)
                      if f is not None]
        if all(back_refls) or not any(back_refls):
            self.back_reflectivity = back_refls[0]
        else:
            raise ValueError("Cannot mix front and back reflectivity measurements")

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

    @property
    def calc_Q(self):
        for x in self.pp, self.pm, self.mp, self.mm:
            if x is not None:
                return x.calc_Q
        raise RuntimeError("No polarization cross sections")

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
        for Ri,xs in zip(R,(self.pp, self.pm, self.mp, self.mm)):
            if xs is not None:
                xs.apply_beam(Ri, resolution)

    def scattering_factors(self, material):
        # doc string is inherited from parent (see below)
        rho, irho, rho_incoh = nsf.neutron_sld(material,
                                               wavelength=self._sf_L,
                                               density=1)
        return rho, irho[self._sf_idx], rho_incoh
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def __len__(self):
        return len(self.calc_Q)

    def plot(self, view=None, **kwargs):
        """
        Plot theory against data.

        Need substrate/surface for Fresnel reflectivity
        """
        view = view if view is not None else self.view

        if view is None: view = Probe.view  # Default to Probe.view

        if view == 'linear':
            self.plot_linear(**kwargs)
        elif view == 'log':
            self.plot_log(**kwargs)
        elif view == 'fresnel':
            self.plot_fresnel(**kwargs)
        elif view == 'q4':
            self.plot_Q4(**kwargs)
        elif view == 'residuals':
            self.plot_residuals(**kwargs)
        elif view == 'SA':
            self.plot_SA(**kwargs)
        elif view == 'resolution':
            self.plot_resolution(**kwargs)
        else:
            raise TypeError("incorrect reflectivity view '%s'"%view)

    def plot_resolution(self, **kwargs):
        self._xs_plot('plot_resolution', **kwargs)
    def plot_linear(self, **kwargs):
        self._xs_plot('plot_linear', **kwargs)
    def plot_log(self, **kwargs):
        self._xs_plot('plot_log', **kwargs)
    def plot_fresnel(self, **kwargs):
        self._xs_plot('plot_fresnel', **kwargs)
    def plot_Q4(self, **kwargs):
        self._xs_plot('plot_Q4', **kwargs)
    def plot_residuals(self, **kwargs):
        self._xs_plot('plot_residuals', **kwargs)
    def plot_SA(self, theory=None, **kwargs):
        import pylab
        if self.pp is None or self.mm is None:
            raise TypeError("cannot form spin asymmetry plot with ++ and --")

        isheld = pylab.ishold()
        pp,mm = self.pp,self.mm
        if hasattr(pp,'R'):
            Q,SA,dSA = spin_asymmetry(pp.Q,pp.R,pp.dR,mm.Q,mm.R,mm.dR)
            if dSA is not None:
                pylab.errorbar(Q, SA, yerr=dSA, xerr=pp.dQ, fmt='.',
                               label=pp.name('data'))
            else:
                pylab.plot(Q,SA,'.',label=pp.name('data'))
            pylab.hold()
        if theory is not None:
            Q,pp,pm,mp,mm = theory
            Q,SA,_ = spin_asymmetry(Q,pp,None,Q,mm,None)
            pylab.plot(Q, SA,label=pp.name('theory'))
        pylab.hold(isheld)
        pylab.xlabel(r'Q (\AA^{-1})')
        pylab.ylabel(r'spin asymmetry $(R^+ -\, R^-) / (R^+ +\, R^-)$')

    def _xs_plot(self, plotter, theory=None, **kwargs):
        import pylab
        # Plot available cross sections
        isheld = pylab.ishold()
        if theory is not None:
            Q,pp,pm,mp,mm = theory
            theory = ((Q,pp),(Q,pm),(Q,mp),(Q,pm))
        else:
            theory = (None,None,None,None)
        for xs,xstheory,suffix in zip((self.pp, self.pm, self.mp, self.mm),
                                      theory,
                                      ('++','+-','-+','--')):
            if xs is not None:
                fn = getattr(xs, plotter)
                fn(theory=xstheory, suffix=suffix, **kwargs)
                pylab.hold(True)
        if not isheld: pylab.hold(False)

    def name(self): return self.pp.name()

def spin_asymmetry(Qp,Rp,dRp,Qm,Rm,dRm):
    r"""
    Compute spin asymmetry for R++, R--.

    **Parameters:**

    *Qp*, *Rp*, *dRp* : vector
        Measured ++ cross section and uncertainty.
    *Qm*, *Rm*, *dRm* : vector
        Measured -- cross section and uncertainty.

    If *dRp*, *dRm* are None then the returned uncertainty will also be None.

    **Returns:**

    *Q*, *SA*, *dSA* : vector
        Computed spin asymmetry and uncertainty.

    **Algorithm:**

    Spin asymmetry, $S_A$, is:

    .. math::

        S_A = \frac{R_{++} - R_{--}}{R_{++} + R_{--}}

    Uncertainty $\Delta S_A$ follows from propagation of error:

    .. math::

        \Delta S_A^2 = \frac{4(R_{++}^2\Delta R_{--}^2-R_{--}^2\Delta R_{++})}
                            {(R_{++} + R_{--})^4}

    """
    Rm = numpy.interp(Qp,Qm,Rm)
    v = (Rp-Rm)/(Rp+Rm)
    if dRp is not None:
        dRm = numpy.interp(Qp,Qm,dRm)
        dvsq = 4 * ((Rp*dRm)**2 - (Rm*dRp)**2) / (Rp+Rm)**4
        dvsq[dvsq<0] = 0
        return Qp, v, sqrt(dvsq)
    else:
        return Qp, v, None




class ProbeSet(Probe):
    def __init__(self, probes):
        self.probes = list(probes)
        self.R = numpy.hstack(p.R for p in self.probes)
        self.dR = numpy.hstack(p.dR for p in self.probes)
        self.dQ = numpy.hstack(p.dQ for p in self.probes)
        self._len = sum([len(p) for p in self.probes])

        back_refls = [f.back_reflectivity for f in self.probes]
        if all(back_refls) or not any(back_refls):
            self.back_reflectivity = back_refls[0]
        else:
            raise ValueError("Cannot mix front and back reflectivity measurements")

    def parameters(self):
        return [p.parameters() for p in self.probes]
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
    def apply_beam(self, calc_Q, calc_R, resolution=True, **kw):
        if not resolution:
            raise UnimplementedError("apply_beam without resolution not supported on ProbeSet; use dQ=0 instead")
        result = [p.apply_beam(calc_Q, calc_R, **kw) for p in self.probes]
        Q,R = [numpy.hstack(v) for v in zip(*result)]
        return Q,R
    def plot(self, theory=None, **kw):
        import pylab
        pylab.clf()
        pylab.hold(True)
        for p,th in self._plotparts(theory): p.plot(theory=th, **kw)
        pylab.hold(False)
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
    def plot_residuals(self, theory=None, **kw):
        for p,th in self._plotparts(theory): p.plot_residuals(theory=th, **kw)
    plot_residuals.__doc__ = Probe.plot_residuals.__doc__
    def _plotparts(self, theory):
        if theory == None:
            for p in self.probes:
                yield p,None
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

    def name(self): return self.probes[0].name()

    def stitch(self, same_Q=0.001, same_dQ=0.001):
        r"""
        Stitch together multiple datasets into a single dataset.

        Points within *tol* of each other and with the same resolution
        are combined by interpolating them to a common $Q$ value then averaged
        using Gaussian error propagation.

        :Returns: probe | Probe
            Combined data set.

        :Algorithm:

        To interpolate a set of points to a common value, first find the
        common $Q$ value:

        .. math::

            \hat Q = \sum{Q_k} / n

        Then for each dataset $k$, find the interval $[i,i+1]$ containing the
        value $Q$, and use it to compute interpolated value for $R$:

        .. math::

            w &= (\hat Q - Q_i)/(Q_{i+1} - Q_i) \\
            \hat R &= w R_{i+1} + (1-w) R_{i+1} \\
            \hat \sigma_{R} &=
                \sqrt{ w^2 \sigma^2_{R_i} + (1-w)^2 \sigma^2_{R_{i+1}} } / n

        Average the resulting $R$ using Gaussian error propagation:

        .. math::

            \hat R &= \sum{\hat R_k}/n \\
            \hat \sigma_R &= \sqrt{\sum \hat \sigma_{R_k}^2}/n

        """
        Q,dQ,R,dR = stitch(self.probes)
        Po = self.probes[0]
        return QProbe(Q,dQ,data=(R,dR),
                      intensity=Po.intensity,
                      background=Po.background,
                      back_absorption=Po.back_absorption,
                      back_reflectivity=Po.back_reflectivity)

class QProbe(Probe):
    """
    A pure Q,R probe

    This probe with no possibility of tricks such as looking up the
    scattering length density based on wavelength, or adjusting for
    alignment errors.
    """
    def __init__(self, Q, dQ, data=None,
                 intensity=1, background=0, back_absorption=1,
                 back_reflectivity=False):
        self.intensity = Parameter.default(intensity,name="intensity")
        self.background = Parameter.default(background,name="background",
                                            limits=[0,inf])
        self.back_absorption = Parameter.default(back_absorption,
                                                 name="back_absorption",
                                                 limits=[0,1])
        self.theta_offset = Constant(0,name="theta_offset")

        self.back_reflectivity = back_reflectivity


        self.Qo, self.dQ = Q, dQ
        if data is not None:
            R,dR = data
        else:
            R,dR = None,None

        self.Qo, self.dQ = Q,dQ
        self.Ro = self.R = R
        self.dR = R

class QPolarizedNeutronProbe(PolarizedNeutronProbe):
    def __init__(self, xs=None):
        self.pp, self.pm, self.mp, self.mm = xs
        self.Q, self.dQ = Qmeasurement_union(xs)
        self._check()
