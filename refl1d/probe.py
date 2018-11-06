# coding=utf-8
# This program is in the public domain
# Author: Paul Kienzle
r"""
Experimental probe.

The experimental probe describes the incoming beam for the experiment.
Scattering properties of the sample are dependent on the type and
energy of the radiation.

See :ref:`data-guide` for details.

"""
from __future__ import with_statement, division, print_function

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
#   Since a given Q, dQ has multiple T, dT, L, dL, oversampling is going
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

import os
import json

import numpy
from numpy import sqrt, pi, inf, sign, log

from periodictable import nsf, xsf
from bumps.parameter import Parameter, Constant
from bumps.plotutil import coordinated_colors, auto_shift
from bumps.data import parse_multi, strip_quotes

from . import fresnel
from .material import Vacuum
from .resolution import QL2T, QT2L, TL2Q, dQdL2dT, dQdT2dLoL, dTdL2dQ
from .resolution import sigma2FWHM, FWHM2sigma
from .stitch import stitch
from .reflectivity import convolve

PROBE_KW = ('T', 'dT', 'L', 'dL', 'data', 'name', 'filename',
            'intensity', 'background', 'back_absorption', 'sample_broadening',
            'theta_offset', 'back_reflectivity', 'data')


def make_probe(**kw):
    """
    Return a reflectometry measurement object of the given resolution.
    """
    radiation = kw.pop('radiation')
    kw = dict((k, v) for k, v in kw.items() if k in PROBE_KW)
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

    A measurement point consists of incident angle, angular resolution,
    incident wavelength, FWHM wavelength resolution, reflectivity and
    uncertainty in reflectivity.

    A probe is a set of points, defined by vectors for point attribute.  For
    convenience, the attribute can be initialized with a scalar if it is
    constant throughout the measurement, but will be set to a vector in
    the probe.  The attributes are initialized as follows:

        *T* : float or [float] | degrees
            Incident angle
        *dT* : float or [float] | degrees
            FWHM angular resolution
        *L* : float or [float] | |Ang|
            Incident wavelength
        *dL* : float or [float] | |Ang|
            FWHM wavelength dispersion
        *data* : ([float], [float])
            R, dR reflectivity measurement and uncertainty

    Measurement properties:

        *intensity* : float or Parameter
           Beam intensity
        *background* : float or Parameter
           Constant background
        *back_absorption* : float or Parameter
           Absorption through the substrate relative to beam intensity.
           A value of 1.0 means complete transmission; a value of 0.0
           means complete absorption.
        *theta_offset* : float or Parameter
           Offset of the sample from perfect alignment
        *sample_broadening* : float or Parameter
           Additional angular divergence from sample curvature.  Should be
           expressed as FWHM.  Scale by sqrt(8 ln 2) ~ 2.35
           to convert from rms to FWHM.
        *back_reflectivity* : True or False
           True if the beam enters through the substrate

    Measurement properties are fittable parameters.  *theta_offset* in
    particular should be set using *probe.theta_offset.dev(dT)*, with *dT*
    equal to the FWHM uncertainty in the peak position for the rocking curve,
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

        *view* : string
            One of 'fresnel', 'logfresnel', 'log', 'linear', 'q4', 'residuals'
        *plot_shift* : float
            The number of pixels to shift each new dataset so
            datasets can be seen separately
        *residuals_shift* :
            The number of pixels to shift each new set of residuals
            so the residuals plots can be seen separately.

    Normally *view* is set directly in the class rather than the
    instance since it is not specific to the view.  Fresnel and Q4
    views are corrected for background and intensity; log and
    linear views show the uncorrected data.
    """
    polarized = False
    Aguide = 270  # default guide field for unpolarized measurements
    view = "fresnel"
    plot_shift = 0
    residuals_shift = 0

    def __init__(self, T=None, dT=0, L=None, dL=0, data=None,
                 intensity=1, background=0, back_absorption=1, theta_offset=0,
                 sample_broadening=0,
                 back_reflectivity=False, name=None, filename=None):
        if T is None or L is None:
            raise TypeError("T and L required")
        if not name and filename:
            name = os.path.splitext(os.path.basename(filename))[0]
        qualifier = " "+name if name is not None else ""
        self.intensity = Parameter.default(intensity,
                                           name="intensity"+qualifier)
        self.background = Parameter.default(background,
                                            name="background"+qualifier,
                                            limits=[0., inf])
        self.back_absorption = Parameter.default(back_absorption,
                                                 name="back_absorption"+qualifier,
                                                 limits=[0., 1.])
        self.theta_offset = Parameter.default(theta_offset,
                                              name="theta_offset"+qualifier)
        self.sample_broadening = Parameter.default(sample_broadening,
                                              name="sample_broadening"+qualifier)
        self.back_reflectivity = back_reflectivity
        if data is not None:
            R, dR = data
        else:
            R, dR = None, None

        self._set_TLR(T, dT, L, dL, R, dR)
        self.name = name
        self.filename = filename

    def _set_TLR(self, T, dT, L, dL, R, dR):
        #if L is None:
        #    L = xsf.xray_wavelength(E)
        #    dL = L * dE/E
        #else:
        #    E = xsf.xray_energy(L)
        #    dE = E * dL/L

        Q = TL2Q(T=T, L=L)
        dQ = dTdL2dQ(T=T, dT=dT + self.sample_broadening.value, L=L, dL=dL)

        # Make sure that we are dealing with vectors
        T, dT, L, dL = [numpy.ones_like(Q)*v for v in (T, dT, L, dL)]

        # Probe stores sorted values for convenience of resolution calculator
        idx = numpy.argsort(Q)
        self.T, self.dT = T[idx], dT[idx]
        self.L, self.dL = L[idx], dL[idx]
        self.Qo, self.dQo = Q[idx], dQ[idx]
        if R is not None:
            R = R[idx]
        if dR is not None:
            dR = dR[idx]
        self.Ro = self.R = R
        self.dR = dR

        # By default the calculated points are the measured points.  Use
        # oversample() for a more accurate resolution calculations.
        self._set_calc(self.T, self.L)

    @staticmethod
    def alignment_uncertainty(w, I, d=0):
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
        if self.Ro is not None:
            self.Ro = 10**self.Ro
            if self.dR is not None:
                self.dR = self.Ro * self.dR * log(10)
            self.R = self.Ro

    def resynth_data(self):
        """
        Generate new data according to the model R ~ N(Ro, dR).

        The resynthesis step is a precursor to refitting the data, as is
        required for certain types of monte carlo error analysis.
        """
        self.R = self.Ro + numpy.random.randn(*self.Ro.shape)*self.dR

    def restore_data(self):
        """
        Restore the original data.
        """
        self.R = self.Ro

    def simulate_data(self, theory, noise=None):
        """
        Set the data for the probe to R, adding random noise dR.

        If noise is None, then use the uncertainty in the probe.

        As a hack, if noise<0, use the probe uncertainty but don't add
        noise to the data.  Don't depend on this behavior.
        """
        self.Ro = theory[1]+0.

        if numpy.isscalar(noise) and noise < 0:
            # leave the probe uncertainty alone, and don't add noise to the data
            self.R = self.Ro
            return

        if noise is None:
            pass
        else:
            self.dR = numpy.asarray(noise)
            if len(self.dR.shape) == 0:  # noise is a scalar
                self.dR = 0.01 * noise * self.Ro
            self.dR[self.dR==0] = 1e-11

        # Add noise to the theory function
        self.resynth_data()

        # Pretend the noisy theory function is the underlying measured data
        # This allows us to resynthesize later, as needed.
        self.Ro = self.R

    def write_data(self, filename,
                   columns=('Q', 'R', 'dR'),
                   header=None):
        """
        Save the data to a file.

        *header* is a string with trailing \\n containing the file header.
        *columns* is a list of column names from Q, dQ, R, dR, L, dL, T, dT.

        The default is to write Q, R, dR data.
        """
        if header is None:
            header = "# %s\n"%' '.join(columns)
        with open(filename, 'w') as fid:
            fid.write(header)
            data = numpy.vstack([getattr(self, c) for c in columns])
            numpy.savetxt(fid, data.T)

    def _set_calc(self, T, L):
        Q = TL2Q(T=T, L=L)

        idx = numpy.argsort(Q)
        self.calc_T = T[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self.unique_L = numpy.unique(self.calc_L)
        self._L_idx = numpy.searchsorted(self.unique_L, L)

    @property
    def Q(self):
        if self.theta_offset.value != 0:
            Q = TL2Q(T=self.T+self.theta_offset.value, L=self.L)
            # TODO: this may break the Q order on measurements with varying L
        else:
            Q = self.Qo
        return Q

    @Q.setter
    def Q(self, Q):
        # If we explicity set Q, then forget what we know about T and L.
        # This will cause theta offset != 0 to fail.
        if hasattr(self, 'T'):
            del self.T, self.L
        self.Qo = Q

    @property
    def dQ(self):
        if self.sample_broadening.value != 0:
            dQ = dTdL2dQ(T=self.T, dT=self.dT + self.sample_broadening.value,
                         L=self.L, dL=self.dL)
        else:
            dQ = self.dQo
        return dQ

    @dQ.setter
    def dQ(self, dQ):
        # If we explicity set dQ, then forget what we know about dT and dL.
        # This will cause sample broadening != 0 to fail.
        if hasattr(self, 'dT'):
            del self.dT, self.dL
        self.dQo = dQ

    @property
    def calc_Q(self):
        if self.theta_offset.value != 0:
            Q = TL2Q(T=self.calc_T+self.theta_offset.value, L=self.calc_L)
            # TODO: this may break the Q order on measurements with varying L
        else:
            Q = self.calc_Qo
        return Q if not self.back_reflectivity else -Q

    def parameters(self):
        return {
            'intensity': self.intensity,
            'background': self.background,
            'back_absorption': self.back_absorption,
            'theta_offset': self.theta_offset,
            'sample_broadening': self.sample_broadening
            }

    def to_dict(self):
        """ Return a dictionary representation of the parameters """
        return dict(type=type(self).__name__,
                    intensity=self.intensity.to_dict(),
                    background=self.background.to_dict(),
                    back_absorption=self.back_absorption.to_dict(),
                    theta_offset=self.theta_offset.to_dict(),
                    sample_broadening=self.sample_broadening.to_dict())

    def scattering_factors(self, material, density):
        """
        Returns the scattering factors associated with the material given
        the range of wavelengths/energies used in the probe.
        """
        raise NotImplementedError

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
        # Assumes self contains sorted Qo and associated T, L
        # Note: calc_Qo is already sorted
        Q = numpy.arange(self.Qo[0], self.Qo[-1], dQ)
        idx = numpy.unique(numpy.searchsorted(self.Qo, Q))
        #print len(idx), len(self.Qo)

        self.T, self.dT = self.T[idx], self.dT[idx]
        self.L, self.dL = self.L[idx], self.dL[idx]
        self.Qo, self.dQo = self.Qo[idx], self.dQo[idx]
        if self.R is not None:
            self.Ro = self.R = self.R[idx]
        if self.dR is not None:
            self.dR = self.dR[idx]
        self._set_calc(self.T, self.L)

    def resolution_guard(self):
        r"""
        Make sure each measured $Q$ point has at least 5 calculated $Q$
        points contributing to it in the range $[-3\Delta Q, 3\Delta Q]$.

        *Not Implemented*
        """
        raise NotImplementedError
        # TODO: implement resolution guard.

    def critical_edge(self, substrate=None, surface=None,
                      n=51, delta=0.25):
        r"""
        Oversample points near the critical edge.

        The critical edge is defined by the difference in scattering
        potential for the *substrate* and *surface* materials, or the
        reverse if *back_reflectivity* is true.

        *n* is the number of $Q$ points to compute near the critical edge.

        *delta* is the relative uncertainty in the material density,
        which defines the range of values which are calculated.

        The $n$ points $Q_i$ are evenly distributed around the critical
        edge in $Q_c \pm \delta Q_c$ by varying angle $\theta$ for a
        fixed wavelength $< \lambda >$, the average of all wavelengths
        in the probe.

        Specifically:

        .. math::

            Q_c^2 &= 16 \pi (\rho - \rho_\text{incident}) \\
            Q_i &= Q_c - \delta_i Q_c (i - (n-1)/2)
                \qquad \text{for} \; i \in 0 \ldots n-1 \\
            \lambda_i &= < \lambda > \\
            \theta_i &= \sin^{-1}(Q_i \lambda_i / 4 \pi)

        If $Q_c$ is imaginary, then $-|Q_c|$ is used instead, so this
        routine can be used for reflectivity signals which scan from
        back reflectivity to front reflectivity.  For completeness,
        the angle $\theta = 0$ is added as well.
        """
        Srho, Sirho = (0, 0) if substrate is None else substrate.sld(self)[:2]
        Vrho, Virho = (0, 0) if surface is None else surface.sld(self)[:2]
        drho = Srho-Vrho if not self.back_reflectivity else Vrho-Srho
        Q_c = sign(drho)*sqrt(16*pi*abs(drho)*1e-6)
        Q = numpy.linspace(Q_c*(1 - delta), Q_c*(1+delta), n)
        L = numpy.average(self.L)
        T = QL2T(Q=Q, L=L)
        T = numpy.hstack((self.T, T, 0))
        L = numpy.hstack((self.L, [L]*(n+1)))
        #print Q
        self._set_calc(T, L)

    def oversample(self, n=20, seed=1):
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
        bias from uniform Q steps.  Depending on the problem, a value of
        *n* between 20 and 100 should lead to stable values for the convolved
        reflectivity.
        """
        if n <= 5:
            raise ValueError("Oversampling with n<=5 is not useful")

        rng = numpy.random.RandomState(seed=seed)
        T = rng.normal(self.T[:, None], self.dT[:, None], size=(len(self.dT), n-1))
        L = rng.normal(self.L[:, None], self.dL[:, None], size=(len(self.dL), n-1))
        T = numpy.hstack((self.T, T.flatten()))
        L = numpy.hstack((self.L, L.flatten()))
        self._set_calc(T, L)

    def _apply_resolution(self, Qin, Rin, interpolation):
        """
        Apply the instrument resolution function
        """
        Q, dQ = _interpolate_Q(self.Q, self.dQ, interpolation)
        R = convolve(Qin, Rin, Q, dQ)
        return Q, R

    def apply_beam(self, calc_Q, calc_R, resolution=True, interpolation=0):
        """
        Apply factors such as beam intensity, background, backabsorption,
        resolution to the data.
        """
        # Note: in-place vector operations are not notably faster.

        # Handle absorption through the substrate, which occurs when Q<0
        # (condition)*C is C when condition is True or 0 when False,
        # (condition)*(C-1)+1 is C when condition is True or 1 when False.
        back = (calc_Q < 0)*(self.back_absorption.value-1)+1
        calc_R = calc_R * back

        # For back reflectivity, reverse the sign of Q after computing
        if self.back_reflectivity:
            calc_Q = -calc_Q
        if calc_Q[-1] < calc_Q[0]:
            calc_Q, calc_R = [v[::-1] for v in (calc_Q, calc_R)]
        if resolution:
            Q, R = self._apply_resolution(calc_Q, calc_R, interpolation)
        else:
            # Given that the target Q points should be in the set of
            # calculated Q values, interp will give us the
            # values of Q at the appropriate R, even if there are
            # guard values, or otherwise some mixture of calculated
            # Q values.  The cost of doing so is going to be n log n
            # in the size of Q, which is a bit pricey, but let's see
            # if it is a problem before optimizing.
            Q, dQ = _interpolate_Q(self.Q, self.dQ, interpolation)
            Q, R = self.Q, numpy.interp(Q, calc_Q, calc_R)
        R = self.intensity.value*R + self.background.value
        #return calc_Q, calc_R
        return Q, R

    def fresnel(self, substrate=None, surface=None):
        """
        Returns a Fresnel reflectivity calculator given the surface and
        and substrate.  The calculated reflectivity includes The Fresnel
        reflectivity for the probe reflecting from a block of material with
        the given substrate.

        Returns F = R(probe.Q), where R is magnitude squared reflectivity.
        """
        # Doesn't use ProbeCache, but this routine is not time critical
        Srho, Sirho = (0, 0) if substrate is None else substrate.sld(self)[:2]
        Vrho, Virho = (0, 0) if surface is None else surface.sld(self)[:2]
        if self.back_reflectivity:
            Srho, Vrho = Vrho, Srho
            Sirho, Virho = Virho, Sirho
        if Srho == Vrho:
            Srho = Vrho + 1
        #I = numpy.ones_like(self.Q)
        I = 1
        calculator = fresnel.Fresnel(rho=Srho*I, irho=Sirho*I,
                                     Vrho=Vrho*I, Virho=Virho*I)
        return calculator

    def save(self, filename, theory, substrate=None, surface=None):
        """
        Save the data and theory to a file.
        """
        fresnel = self.fresnel(substrate, surface)
        Q, R = theory
        fid = open(filename, "w")
        fid.write("# intensity: %.15g\n# background: %.15g\n"
                  % (self.intensity.value, self.background.value))
        if len(Q) != len(self.Q):
            # Saving interpolated data
            A = numpy.array((Q, R, fresnel(Q)))
            fid.write("# %17s %20s %20s\n"
                      % ("Q (1/A)", "theory", "fresnel"))
        elif getattr(self, 'R', None) is not None:
            A = numpy.array((self.Q, self.dQ, self.R, self.dR,
                             R, fresnel(self.Q)))
            fid.write("# %17s %20s %20s %20s %20s %20s\n"
                      % ("Q (1/A)", "dQ (1/A)", "R", "dR", "theory", "fresnel"))
        else:
            A = numpy.array((self.Q, self.dQ, R, fresnel(self.Q)))
            fid.write("# %17s %20s %20s %20s\n"
                      % ("Q (1/A)", "dQ (1/A)", "theory", "fresnel"))
        #print "A", self.Q.shape, A.shape
        numpy.savetxt(fid, A.T, fmt="%20.15g")
        fid.close()

    def plot(self, view=None, **kwargs):
        """
        Plot theory against data.

        Need substrate/surface for Fresnel-normalized reflectivity
        """

        view = view if view is not None else self.view

        if view == 'linear':
            self.plot_linear(**kwargs)
        elif view == 'log':
            self.plot_log(**kwargs)
        elif view == 'fresnel':
            self.plot_fresnel(**kwargs)
        elif view == 'logfresnel':
            self.plot_logfresnel(**kwargs)
        elif view == 'q4':
            self.plot_Q4(**kwargs)
        elif view == 'resolution':
            self.plot_resolution(**kwargs)
        elif view.startswith('resid'):
            self.plot_residuals(**kwargs)
        elif view == 'fft':
            self.plot_fft(**kwargs)
        elif view == 'SA': # SA uses default plot
            self.plot(view=None, **kwargs)
        else:
            raise TypeError("incorrect reflectivity view '%s'"%view)


    def plot_resolution(self, suffix='', label=None, **kwargs):
        import pylab
        pylab.plot(self.Q, self.dQ,
                   label=self.label(prefix=label, suffix=suffix))
        pylab.xlabel(r'Q ($\AA^{-1}$)')
        pylab.ylabel(r'Q resolution ($1-\sigma \AA^{-1}$)')
        pylab.title('Measurement resolution')


    def plot_linear(self, **kwargs):
        """
        Plot the data associated with probe.
        """
        import pylab
        self._plot_pair(ylabel='Reflectivity', **kwargs)
        pylab.yscale('linear')

    def plot_log(self, **kwargs):
        """
        Plot the data associated with probe.
        """
        import pylab
        self._plot_pair(ylabel='Reflectivity', **kwargs)
        pylab.yscale('log')

    def plot_logfresnel(self, *args, **kw):
        """
        Plot the log Fresnel-normalized reflectivity associated with the probe.
        """
        import pylab
        self.plot_fresnel(*args, **kw)
        pylab.yscale('log')

    def plot_fresnel(self, substrate=None, surface=None, **kwargs):
        r"""
        Plot the Fresnel-normalized reflectivity associated with the probe.

        Note that the Fresnel reflectivity has the intensity and background
        applied before normalizing so that hydrogenated samples display more
        cleanly.  The formula to reproduce the graph is:

        .. math::

             R' = R / (F(Q) I + B)
             \Delta R' = \Delta R / (F(Q) I + B)

        where $I$ is the intensity and $B$ is the background.
        """
        if substrate is None and surface is None:
            raise TypeError("Fresnel-normalized reflectivity needs substrate or surface")
        F = self.fresnel(substrate=substrate, surface=surface)
        #print("substrate", substrate, "surface", surface)
        def scale(Q, dQ, R, dR):
            Q, fresnel = self.apply_beam(self.calc_Q, F(self.calc_Q))
            return Q, dQ, R/fresnel, dR/fresnel
        if substrate is None:
            name = "air:%s" % surface.name
        elif surface is None or isinstance(surface, Vacuum):
            name = substrate.name
        else:
            name = "%s:%s" % (substrate.name, surface.name)
        self._plot_pair(correct=scale, ylabel='R/(R(%s)' % name, **kwargs)

    def plot_Q4(self, **kwargs):
        r"""
        Plot the Q**4 reflectivity associated with the probe.

        Note that Q**4 reflectivity has the intensity and background applied
        so that hydrogenated samples display more cleanly.  The formula
        to reproduce the graph is:

        .. math::

             R' = R / ( (100*Q)^{-4} I + B)
             \Delta R' = \Delta R / ( (100*Q)^{-4} I + B )

        where $B$ is the background.
        """
        scale = lambda Q, dQ, R, dR: (
            Q, dQ,
            #R/numpy.maximum(1e-8*Q**-4, self.background.value),
            #dR/numpy.maximum(1e-8*Q**-4, self.background.value))
            R/(1e-8*Q**-4*self.intensity.value + self.background.value),
            dR/(1e-8*Q**-4*self.intensity.value + self.background.value))
        #Q4[Q4==0] = 1
        self._plot_pair(correct=scale, ylabel='R (100 Q)^4', **kwargs)

    def _plot_pair(self, theory=None,
                   correct=lambda Q, dQ, R, dR: (Q, dQ, R, dR),
                   ylabel="", suffix="", label=None,
                   plot_shift=None, **kw):
        import pylab
        c = coordinated_colors()
        plot_shift = plot_shift if plot_shift is not None else Probe.plot_shift
        trans = auto_shift(plot_shift)
        if hasattr(self, 'R') and self.R is not None:
            Q, dQ, R, dR = correct(self.Q, self.dQ, self.R, self.dR)
            pylab.errorbar(Q, R, yerr=dR, xerr=dQ, capsize=0,
                           fmt='.', color=c['light'], transform=trans,
                           label=self.label(prefix=label,
                                            gloss='data',
                                            suffix=suffix))
        if theory is not None:
            Q, R = theory
            Q, dQ, R, dR = correct(Q, 0, R, 0)
            pylab.plot(Q, R, '-',
                       color=c['dark'], transform=trans,
                       label=self.label(prefix=label,
                                        gloss='theory',
                                        suffix=suffix))
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel(ylabel)
        #pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        h = pylab.legend(fancybox=True, numpoints=1)
        h.get_frame().set_alpha(0.5)

    def plot_residuals(self, theory=None, suffix='', label=None,
                       plot_shift=None, **kwargs):
        import pylab
        plot_shift = plot_shift if plot_shift is not None else Probe.residuals_shift
        trans = auto_shift(plot_shift)
        if theory is not None and self.R is not None:
            c = coordinated_colors()
            Q, R = theory
            residual = (R - self.R)/self.dR
            pylab.plot(self.Q, residual,
                       '.', color=c['light'],
                       transform=trans,
                       label=self.label(prefix=label, suffix=suffix))
        pylab.axhline(1, color='black', ls='--', lw=1)
        pylab.axhline(0, color='black', lw=1)
        pylab.axhline(-1, color='black', ls='--', lw=1)
        pylab.xlabel('Q (inv A)')
        pylab.ylabel('(theory-data)/error')
        pylab.legend(numpoints=1)

    def plot_fft(self, theory=None, suffix='', label=None,
                 substrate=None, surface=None, **kwargs):
        """
        FFT analysis of reflectivity signal.
        """
        raise NotImplementedError
        import pylab
        c = coordinated_colors()
        if substrate is None and surface is None:
            raise TypeError("FFT reflectivity needs substrate or surface")
        F = self.fresnel(substrate=substrate, surface=surface)
        #Qc = sqrt(16*pi*substrate)
        Qc = 0
        Qmax = max(self.Q)
        T = numpy.linspace(Qc, Qmax, len(self.Q))
        z = numpy.linspace(0, 2*pi/Qmax, len(self.Q)//2)
        if hasattr(self, 'R'):
            signal = numpy.interp(T, self.Q, self.R/F(self.Q))
            A = abs(numpy.fft.fft(signal - numpy.average(signal)))
            A = A[:len(A)//2]
            pylab.plot(z, A, '.-', color=c['light'],
                       label=self.label(prefix=label,
                                        gloss='data',
                                        suffix=suffix))
        if theory is not None:
            Q, R = theory
            signal = numpy.interp(T, Q, R/F(Q))
            A = abs(numpy.fft.fft(signal-numpy.average(signal)))
            A = A[:len(A)//2]
            pylab.plot(z, A, '-', color=c['dark'],
                       label=self.label(prefix=label,
                                        gloss='theory',
                                        suffix=suffix))
        pylab.xlabel('w (A)')
        if substrate is None:
            name = "air:%s" % surface.name
        elif surface is None or isinstance(surface, Vacuum):
            name = substrate.name
        else:
            name = "%s:%s" % (substrate.name, surface.name)
        pylab.ylabel('|FFT(R/R(%s))|' % name)

    def label(self, prefix=None, gloss="", suffix=""):
        if not prefix:
            prefix = self.name
        if prefix:
            return " ".join((prefix+suffix, gloss)) if gloss else prefix
        else:
            return suffix+" "+gloss if gloss else None

class XrayProbe(Probe):
    """
    X-Ray probe.

    By providing a scattering factor calculator for X-ray scattering, model
    components can be defined by mass density and chemical composition.
    """
    radiation = "xray"

    def scattering_factors(self, material, density):
        # doc string is inherited from parent (see below)
        # Note: the real density is calculated as a scale factor applied to
        # the returned sld as computed assuming density=1
        rho, irho = xsf.xray_sld(material,
                                 wavelength=self.unique_L,
                                 density=density)
        # TODO: support wavelength dependent systems
        return rho[0], irho[0], 0
        #return rho[self._L_idx], irho[self._L_idx], 0
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__


class NeutronProbe(Probe):
    """
    Neutron probe.

    By providing a scattering factor calculator for X-ray scattering, model
    components can be defined by mass density and chemical composition.
    """
    radiation = "neutron"

    def scattering_factors(self, material, density):
        # doc string is inherited from parent (see below)
        # Note: the real density is calculated as a scale factor applied to
        # the returned sld as computed assuming density=1
        rho, irho, rho_incoh = nsf.neutron_sld(material,
                                               wavelength=self.unique_L,
                                               density=density)
        # TODO: support wavelength dependent systems
        return rho, irho[0], rho_incoh
        #return rho, irho[self._L_idx], rho_incoh
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__


class ProbeSet(Probe):
    def __init__(self, probes, name=None):
        self.probes = list(probes)
        self.R = numpy.hstack(p.R for p in self.probes)
        self.dR = numpy.hstack(p.dR for p in self.probes)
        self.name = name if name is not None else self.probes[0].name

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

    def simulate_data(self, theory, noise=2):
        """
            Simulate data, allowing for noise to be a dR array for each Q point.
        """
        Q, R = theory
        dR = numpy.asarray(noise)
        offset = 0
        for p in self.probes:
            n = len(p.Q)
            if len(self.dR.shape) > 0:
                noise = dR[offset:offset+n]
            p.simulate_data(theory=(Q[offset:offset+n], R[offset:offset+n]),
                            noise=noise)
            offset += n
    simulate_data.__doc__ = Probe.simulate_data.__doc__

    @property
    def Q(self):
        return numpy.hstack(p.Q for p in self.probes)

    @property
    def calc_Q(self):
        return numpy.unique(numpy.hstack(p.calc_Q for p in self.probes))

    @property
    def dQ(self):
        return numpy.hstack(p.dQ for p in self.probes)

    @property
    def unique_L(self):
        return numpy.unique(numpy.hstack(p.unique_L for p in self.probes))

    def oversample(self, **kw):
        for p in self.probes:
            p.oversample(**kw)
    oversample.__doc__ = Probe.oversample.__doc__

    def scattering_factors(self, material, density):
        # TODO: support wavelength dependent systems
        return self.probes[0].scattering_factors(material, density)
        # result = [p.scattering_factors(material, density) for p in self.probes]
        # return [numpy.hstack(v) for v in zip(*result)]
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def apply_beam(self, calc_Q, calc_R, interpolation=0, **kw):
        result = [p.apply_beam(calc_Q, calc_R, **kw) for p in self.probes]
        Q, R = [numpy.hstack(v) for v in zip(*result)]
        return Q, R

    def fresnel(self, *args, **kw):
        return self.probes[0].fresnel(*args, **kw)
    fresnel.__doc__ = Probe.fresnel.__doc__

    def save(self, filename, theory, substrate=None, surface=None):
        for i, (p, th) in enumerate(self.parts(theory=theory)):
            p.save(filename+str(i+1), th, substrate=substrate, surface=surface)
    save.__doc__ = Probe.save.__doc__

    def plot(self, theory=None, **kw):
        import pylab
        for p, th in self.parts(theory):
            p.plot(theory=th, **kw)
    plot.__doc__ = Probe.plot.__doc__

    def plot_resolution(self, **kw):
        for p in self.probes:
            p.plot_resolution(**kw)
    plot_resolution.__doc__ = Probe.plot_resolution.__doc__

    def plot_linear(self, theory=None, **kw):
        for p, th in self.parts(theory):
            p.plot_linear(theory=th, **kw)
    plot_linear.__doc__ = Probe.plot_linear.__doc__

    def plot_log(self, theory=None, **kw):
        for p, th in self.parts(theory):
            p.plot_log(theory=th, **kw)
    plot_log.__doc__ = Probe.plot_log.__doc__

    def plot_fresnel(self, theory=None, **kw):
        for p, th in self.parts(theory):
            p.plot_fresnel(theory=th, **kw)
    plot_fresnel.__doc__ = Probe.plot_fresnel.__doc__

    def plot_logfresnel(self, theory=None, **kw):
        for p, th in self.parts(theory):
            p.plot_logfresnel(theory=th, **kw)
    plot_logfresnel.__doc__ = Probe.plot_logfresnel.__doc__

    def plot_Q4(self, theory=None, **kw):
        for p, th in self.parts(theory):
            p.plot_Q4(theory=th, **kw)
    plot_Q4.__doc__ = Probe.plot_Q4.__doc__

    def plot_residuals(self, theory=None, **kw):
        for p, th in self.parts(theory):
            p.plot_residuals(theory=th, **kw)
    plot_residuals.__doc__ = Probe.plot_residuals.__doc__

    def parts(self, theory):
        if theory is None:
            for p in self.probes:
                yield p, None
        else:
            offset = 0
            Q, R = theory
            for p in self.probes:
                n = len(p.Q)
                yield p, (Q[offset:offset+n], R[offset:offset+n])
                offset += n

    def shared_beam(self, intensity=1, background=0,
                    back_absorption=1, theta_offset=0,
                    sample_broadening=0):
        """
        Share beam parameters across all segments.

        New parameters are created for *intensity*, *background*,
        *theta_offset*, *sample_broadening* and *back_absorption*
        and assigned to the all segments.  These can be replaced
        with an explicit parameter in an individual segment if that
        parameter is independent.
        """
        intensity = Parameter.default(intensity, name="intensity")
        background = Parameter.default(background,
                                       name="background",
                                       limits=[0, inf])
        back_absorption = Parameter.default(back_absorption,
                                            name="back_absorption",
                                            limits=[0, 1])
        theta_offset = Parameter.default(theta_offset, name="theta_offset")
        sample_broadening = Parameter.default(sample_broadening,
                                              name="sample_broadening",
                                              limits=[0, inf])
        for p in self.probes:
            p.intensity = intensity
            p.background = background
            p.back_absorption = back_absorption
            p.theta_offset = theta_offset
            p.sample_broadening = sample_broadening

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

        Then for each dataset $k$, find the interval $[i, i+1]$ containing the
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
        Q, dQ, R, dR = stitch(self.probes)
        Po = self.probes[0]
        return QProbe(Q, dQ, data=(R, dR),
                      intensity=Po.intensity,
                      background=Po.background,
                      back_absorption=Po.back_absorption,
                      back_reflectivity=Po.back_reflectivity)


def load4(filename, keysep=":", sep=None, comment="#", name=None,
          intensity=1, background=0, back_absorption=1,
          back_reflectivity=False, Aguide=270, H=0,
          theta_offset=0, sample_broadening=0,
          L=None, dL=None, T=None, dT=None,
          FWHM=False, radiation=None,
          columns=None,
         ):
    r"""
    Load in four column data Q, R, dR, dQ.

    The file is loaded with *bumps.data.parse_multi*.  *keysep*
    defaults to ':' so that header data looks like JSON key: value
    pairs.  *sep* is None so that the data uses white-space separated
    columns.  *comment* is the standard '#' comment character, used
    for "# key: value" lines, for commenting out data lines using
    "#number number number number", and for adding comments after
    individual data lines.  The parser isn't very sophisticated, so
    be nice.

    *intensity* is the overall beam intensity, *background* is the
    overall background level, and *back_absorption* is the relative
    intensity of data measured at negative Q compared to positive Q
    data.  These can be values or a bumps *Parameter* objects.

    *back_reflectivity* is True if reflectivity was measured through
    the substrate.  This allows you to arrange the model from substrate
    to surface regardless of whether you are measuring through the
    substrate or reflecting off the surface.

    *theta_offset* indicates sample alignment.  In order to use theta
    offset you need to be able to convert from Q to wavelength and angle
    by providing values for the wavelength or the angle, and the associated
    resolution.

    For monochromatic sources you can supply *L*, *dLoL* when you call *load4*,
    or you can store it in the header of the file::

        # wavelength: 4.75  # Ang
        # wavelength_resolution: 0.02  # Ang (1-sigma)

    For time of flight sources, angle is fixed and wavelength is
    varying, so you can supply *T*, *dT* in degrees when you call *load4*,
    or you can store it in the header of the file::

        # angle: 2  # degrees
        # angular_resolution: 0.2  # degrees (1-sigma)

    If both angle and wavelength are varying in the data, you can specify
    a separate value for each point, such the following::

        # wavelength: [1, 1.2, 1.5, 2.0, ...]
        # wavelength_resolution: [0.02, 0.02, 0.02, ...]

    *sample_broadening* in degrees (1-$\sigma$ rms) adds to the angular_resolution.

    *Aguide* and *H* are parameters for polarized beam measurements
    indicating the magnitude and direction of the applied field.

    Polarized data is represented using a multi-section data file,
    with blank lines separating each section.  Each section must
    have a polarization keyword, with value "++", "+-", "-+" or "--".

    *FWHM* is True if dQ, dT, dL are given as FWHM rather than 1-$\sigma$.
    *dR* is always 1-$\sigma$.  *sample_broadening* is always FWHM.

    *radiation* is 'xray' or 'neutron', depending on whether X-ray or
    neutron scattering length density calculator should be used for
    determining the scattering length density of a material.

    *columns* is a string giving the column order in the file.  Default
    order is "Q R dR dQ".
    """
    data = parse_multi(filename, keysep=keysep, sep=sep, comment=comment)
    if columns:
        actual = columns.split()
        natural = "Q R dR dQ".split()
        order = [natural.index(k) for k in actual]
    else:
        order = [0, 1, 2, 3]
    def _as_Qprobe(data):
        Q, R, dR, dQ = (data[1][k] for k in order)

        if FWHM: # dQ defaults to 1-sigma, if FWHM is not True
            dQ = FWHM2sigma(dQ)

        # support calculation of sld from material based on radiation type
        if radiation is not None:
            data_radiation = radiation
        elif 'radiation' in data[0]:
            data_radiation = json.loads(data[0]['radiation'])
        else:
            data_radiation = None
        if data_radiation == 'xray':
            make_probe = XrayProbe
        elif data_radiation == 'neutron':
            make_probe = NeutronProbe
        else:
            make_probe = Probe

        # Get wavelength from header if it is not provided as an argument
        data_L = data_T = None
        if L is not None:
            data_L = L
        elif 'wavelength' in data[0]:
            data_L = json.loads(data[0]['wavelength'])
        if T is not None:
            data_T = T
        elif 'angle' in data[0]:
            data_T = json.loads(data[0]['angle'])
        if data_L is not None:
            if dL is not None:
                data_dL = dL
            elif 'wavelength_resolution' in data[0]:
                data_dL = json.loads(data[0]['wavelength_resolution'])
            else:
                raise ValueError("Need wavelength_resolution to determine dT")
            data_dL = sigma2FWHM(data_dL) if not FWHM else data_dL
            data_T = QL2T(Q, data_L)
            data_dT = dQdL2dT(Q, dQ, data_L, data_dL)
        elif data_T is not None:
            if dT is not None:
                data_dT = dT
            elif 'angular_resolution' in data[0]:
                data_dT = json.loads(data[0]['angular_resolution'])
            else:
                raise ValueError("Need angular_resolution to determine dL")
            data_dT = sigma2FWHM(data_dT) if not FWHM else data_dT
            data_L = QT2L(Q, data_T)
            data_dLoL = dQdT2dLoL(Q, dQ, data_T, data_dT)
            data_dL = data_dLoL * data_L

        if data_L is not None:
            probe = make_probe(
                T=data_T, dT=data_dT,
                L=data_L, dL=data_dL,
                data=(R, dR),
                name=name,
                filename=filename,
                intensity=intensity,
                background=background,
                back_absorption=back_absorption,
                theta_offset=theta_offset,
                sample_broadening=sample_broadening,
                back_reflectivity=back_reflectivity,
            )
        else:
            probe = QProbe(
                Q, dQ, data=(R, dR),
                name=name,
                filename=filename,
                intensity=intensity,
                background=background,
                back_absorption=back_absorption,
                back_reflectivity=back_reflectivity,
            )
        return probe

    if len(data) == 1:
        probe = _as_Qprobe(data[0])
    else:
        data_by_xs = dict((strip_quotes(d[0]["polarization"]), _as_Qprobe(d))
                          for d in data)
        if not set(data_by_xs.keys()) <= set('-- -+ +- ++'.split()):
            raise ValueError("Unknown cross sections in: "
                             + ", ".join(sorted(data_by_xs.keys())))
        xs = [data_by_xs.get(xs, None) for xs in ('--', '-+', '+-', '++')]

        if any(isinstance(d, QProbe) for d in xs if d is not None):
            probe = PolarizedQProbe(xs, Aguide=Aguide, H=H)
        else:
            probe = PolarizedNeutronProbe(xs, Aguide=Aguide, H=H)
    return probe


class QProbe(Probe):
    """
    A pure Q, R probe

    This probe with no possibility of tricks such as looking up the
    scattering length density based on wavelength, or adjusting for
    alignment errors.
    """
    def __init__(self, Q, dQ, data=None, name=None, filename=None,
                 intensity=1, background=0, back_absorption=1,
                 back_reflectivity=False):
        if not name and filename:
            name = os.path.splitext(os.path.basename(filename))[0]
        qualifier = " "+name if name is not None else ""
        self.intensity = Parameter.default(intensity, name="intensity"+qualifier)
        self.background = Parameter.default(background, name="background"+qualifier,
                                            limits=[0, inf])
        self.back_absorption = Parameter.default(back_absorption,
                                                 name="back_absorption"+qualifier,
                                                 limits=[0, 1])
        self.theta_offset = Constant(0, name="theta_offset"+qualifier)
        self.sample_broadening = Constant(0, name="sample_broadening"+qualifier)

        self.back_reflectivity = back_reflectivity


        if data is not None:
            R, dR = data
        else:
            R, dR = None, None

        self.Q, self.dQ = Q, dQ
        self.Ro = self.R = R
        self.dR = dR
        self.unique_L = None
        self.calc_Qo = self.Qo
        self.name = name


def measurement_union(xs):
    """
    Determine the unique (T, dT, L, dL) across all datasets.
    """

    # First gather a set of unique tuples in angle and wavelength
    TL = set()
    for x in xs:
        if x is not None:
            TL |= set(zip(x.T, x.dT, x.L, x.dL))
    T, dT, L, dL = zip(*[item for item in TL])
    T, dT, L, dL = [numpy.asarray(v) for v in (T, dT, L, dL)]

    # Convert to Q, dQ
    Q = TL2Q(T, L)
    dQ = dTdL2dQ(T, dT, L, dL)

    # Sort by Q
    idx = numpy.argsort(Q)
    T, dT, L, dL, Q, dQ = T[idx], dT[idx], L[idx], dL[idx], Q[idx], dQ[idx]
    if abs(Q[1:] - Q[:-1]).any() < 1e-14:
        raise ValueError("Q is not unique")
    return T, dT, L, dL, Q, dQ

def Qmeasurement_union(xs):
    """
    Determine the unique Q, dQ across all datasets.
    """
    Qset = set()
    for x in xs:
        if x is not None:
            Qset |= set(zip(x.Q, x.dQ))
    Q, dQ = [numpy.array(v) for v in zip(*sorted(Qset))]
    if abs(Q[1:] - Q[:-1]).any() < 1e-14:
        raise ValueError("Q values differ by less than 1e-14")
    return Q, dQ

class PolarizedNeutronProbe(object):
    """
    Polarized neutron probe

    *xs* (4 x NeutronProbe) is a sequence pp, pm, mp and mm.

    *Aguide* (degrees) is the angle of the applied field relative
    to the plane of the sample, with angle 270º in the plane of the sample.

    *H* (tesla) is the magnitude of the applied field
    """
    view = None  # Default to Probe.view so only need to change in one place
    substrate = surface = None
    polarized = True
    def __init__(self, xs=None, name=None, Aguide=270, H=0):
        self._xs = xs

        if name is None and self.xs[0] is not None:
            name = self.xs[0].name
        self.name = name
        self.T, self.dT, self.L, self.dL, self.Q, self.dQ \
            = measurement_union(xs)
        self._set_calc(self.T, self.L)
        self._check()
        spec = " "+name if name else ""
        self.H = Parameter.default(H, name="H"+spec)
        self.Aguide = Parameter.default(Aguide, name="Aguide"+spec,
                                        limits=[-360, 360])
    @property
    def xs(self):
        return self._xs  # Don't let user replace xs

    @property
    def pp(self):
        return self.xs[3]

    @property
    def pm(self):
        return self.xs[2]

    @property
    def mp(self):
        return self.xs[1]

    @property
    def mm(self):
        return self.xs[0]

    def parameters(self):
        mm, mp, pm, pp = [(xsi.parameters() if xsi else None)
                          for xsi in self.xs]
        return {
            'pp': pp, 'pm': pm, 'mp': mp, 'mm': mm,
            'Aguide': self.Aguide, 'H': self.H,
        }

    def to_dict(self):
        """ Return a dictionary representation of the parameters """
        mm, mp, pm, pp = [(xsi.to_dict() if xsi else None)
                          for xsi in self.xs]
        return dict(type=type(self).__name__,
                    pp=pp, pm=pm, mp=mp, mm=mm,
                    a_guide=self.Aguide.to_dict(),
                    h=self.H.to_dict())

    def resynth_data(self):
        for p in self.xs:
            if p is not None:
                p.resynth_data()
    resynth_data.__doc__ = Probe.resynth_data.__doc__

    def restore_data(self):
        for p in self.xs:
            if p is not None:
                p.restore_data()
    restore_data.__doc__ = Probe.restore_data.__doc__

    def simulate_data(self, theory, noise=2):
        if numpy.isscalar(noise):
            noise = [noise]*4
        for data_k, theory_k, noise_k in zip(self.xs, theory, noise):
            if data_k is not None:
                data_k.simulate_data(theory=theory_k, noise=noise_k)
    simulate_data.__doc__ = Probe.simulate_data.__doc__

    def _check(self):
        back_refls = [f.back_reflectivity for f in self.xs if f is not None]
        if all(back_refls) or not any(back_refls):
            self.back_reflectivity = back_refls[0]
        else:
            raise ValueError("Cannot mix front and back reflectivity measurements")

    def shared_beam(self, intensity=1, background=0,
                    back_absorption=1, theta_offset=0,
                    sample_broadening=0):
        """
        Share beam parameters across all four cross sections.

        New parameters are created for *intensity*, *background*,
        *theta_offset*, *sample_broadening* and *back_absorption*
        and assigned to the all cross sections.  These can be replaced
        with an explicit parameter in an individual cross section if that
        parameter is independent.
        """
        intensity = Parameter.default(intensity, name="intensity")
        background = Parameter.default(background, name="background",
                                       limits=[0, inf])
        back_absorption = Parameter.default(back_absorption,
                                            name="back_absorption",
                                            limits=[0, 1])
        theta_offset = Parameter.default(theta_offset, name="theta_offset")
        sample_broadening = Parameter.default(sample_broadening,
                                              name="sample_broadening",
                                              limits=[0, inf])
        for x in self.xs:
            if x is not None:
                x.intensity = intensity
                x.background = background
                x.back_absorption = back_absorption
                x.theta_offset = theta_offset
                x.sample_broadening = sample_broadening

    def oversample(self, n=6, seed=1):
        # doc string is inherited from parent (see below)
        rng = numpy.random.RandomState(seed=seed)
        T = rng.normal(self.T[:, None], self.dT[:, None], size=(len(self.dT), n))
        L = rng.normal(self.L[:, None], self.dL[:, None], size=(len(self.dL), n))
        T = T.flatten()
        L = L.flatten()
        self._set_calc(T, L)
    oversample.__doc__ = Probe.oversample.__doc__

    @property
    def calc_Q(self):
        return self.calc_Qo

    def _set_calc(self, T, L):
        # TODO: shouldn't clone code from probe
        Q = TL2Q(T=T, L=L)

        idx = numpy.argsort(Q)
        self.calc_T = T[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self.unique_L = numpy.unique(self.calc_L)
        self._L_idx = numpy.searchsorted(self.unique_L, L)

    def apply_beam(self, Q, R, resolution=True, interpolation=0):
        """
        Apply factors such as beam intensity, background, backabsorption,
        and footprint to the data.
        """
        return [(xs.apply_beam(Q, Ri, resolution, interpolation) if xs else None)
                for xs, Ri in zip(self.xs, R)]

    def fresnel(self, *args, **kw):
        return self.pp.fresnel(*args, **kw)
    fresnel.__doc__ = Probe.fresnel.__doc__

    def scattering_factors(self, material, density):
        # doc string is inherited from parent (see below)
        rho, irho, rho_incoh = nsf.neutron_sld(material,
                                               wavelength=self.unique_L,
                                               density=density)
        # TODO: support wavelength dependent systems
        #print("sf", str(material), type(rho), type(irho[0]))
        return rho, irho[0], rho_incoh
        #return rho, irho[self._L_idx], rho_incoh
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def select_corresponding(self, theory):
        """
        Select theory points corresponding to the measured data.

        Since we have evaluated theory at every Q, it is safe to interpolate
        measured Q into theory, since it will land on a node,
        not in an interval.
        """

        Qth, Rth = theory
        return [None if x_data is None
                else (x_data.Q, numpy.interp(x_data.Q, Qth, x_th))
                for x_data, x_th in zip(self.xs, Rth)]


    def save(self, filename, theory, substrate=None, surface=None):
        for xsi, xsi_th, suffix in zip(self.xs, theory, ('A', 'B', 'C', 'D')):
            if xsi is not None:
                xsi.save(filename+suffix, xsi_th,
                         substrate=substrate, surface=surface)
    save.__doc__ = Probe.save.__doc__

    def plot(self, view=None, **kwargs):
        """
        Plot theory against data.

        Need substrate/surface for Fresnel-normalized reflectivity
        """
        view = view if view is not None else self.view

        if view is None: view = Probe.view  # Default to Probe.view

        if view == 'linear':
            self.plot_linear(**kwargs)
        elif view == 'log':
            self.plot_log(**kwargs)
        elif view == 'fresnel':
            self.plot_fresnel(**kwargs)
        elif view == 'logfresnel':
            self.plot_logfresnel(**kwargs)
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

    def plot_logfresnel(self, **kwargs):
        self._xs_plot('plot_logfresnel', **kwargs)

    def plot_Q4(self, **kwargs):
        self._xs_plot('plot_Q4', **kwargs)

    def plot_residuals(self, **kwargs):
        self._xs_plot('plot_residuals', **kwargs)

    def plot_SA(self, theory=None, label=None, plot_shift=None,
                **kwargs):
        import pylab
        if self.pp is None or self.mm is None:
            raise TypeError("cannot form spin asymmetry plot without ++ and --")

        plot_shift = plot_shift if plot_shift is not None else Probe.plot_shift
        trans = auto_shift(plot_shift)
        pp, mm = self.pp, self.mm
        c = coordinated_colors()
        if hasattr(pp, 'R') and hasattr(mm, 'R') and pp.R is not None and mm.R is not None:
            Q, SA, dSA = spin_asymmetry(pp.Q, pp.R, pp.dR, mm.Q, mm.R, mm.dR)
            if dSA is not None:
                pylab.errorbar(Q, SA, yerr=dSA, xerr=pp.dQ, fmt='.', capsize=0,
                               label=pp.label(prefix=label, gloss='data'),
                               transform=trans,
                               color=c['light'])
            else:
                pylab.plot(Q, SA, '.',
                           label=pp.label(prefix=label, gloss='data'),
                           transform=trans,
                           color=c['light'])
        if theory is not None:
            mm, mp, pm, pp = theory
            Q, SA, _ = spin_asymmetry(pp[0], pp[1], None, mm[0], mm[1], None)
            pylab.plot(Q, SA,
                       label=self.pp.label(prefix=label, gloss='theory'),
                       transform=trans,
                       color=c['dark'])
        pylab.xlabel(r'Q (\AA^{-1})')
        pylab.ylabel(r'spin asymmetry $(R^{++} -\, R^{--}) / (R^{++} +\, R^{--})$')
        pylab.legend(numpoints=1)

    def _xs_plot(self, plotter, theory=None, **kwargs):
        import pylab
        # Plot available cross sections
        if theory is None:
            theory = (None, None, None, None)
        for x_data, x_th, suffix in zip(self.xs, theory,
                                        ('$^{--}$', '$^{-+}$', '$^{+-}$', '$^{++}$')):
            if x_data is not None:
                fn = getattr(x_data, plotter)
                fn(theory=x_th, suffix=suffix, **kwargs)

def spin_asymmetry(Qp, Rp, dRp, Qm, Rm, dRm):
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

        \Delta S_A^2 = \frac{4(R_{++}^2\Delta R_{--}^2+R_{--}^2\Delta R_{++})}
                            {(R_{++} + R_{--})^4}

    """
    Rm = numpy.interp(Qp, Qm, Rm)
    v = (Rp-Rm)/(Rp+Rm)
    if dRp is not None:
        dRm = numpy.interp(Qp, Qm, dRm)
        dvsq = 4 * ((Rp*dRm)**2 + (Rm*dRp)**2) / (Rp+Rm)**4
        dvsq[dvsq < 0] = 0
        return Qp, v, sqrt(dvsq)
    else:
        return Qp, v, None



def _interpolate_Q(Q, dQ, n):
    """
    Helper function to interpolate between data points.

    *n* is the number of points to show between existing points.
    """
    if n > 0:
        # Extend the Q-range by one point on either side
        Q = numpy.hstack((0.5*(Q[0]-Q[1]), Q, 0.5*(3.*Q[-1]-Q[-2])))
        dQ = numpy.hstack((0.5*(dQ[0]-dQ[1]), dQ, 0.5*(3.*dQ[-1]-dQ[-2])))
        index = numpy.arange(0, len(Q), dtype='d')
        subindex = numpy.arange(0, (n+1)*(len(Q)-1)+1, dtype='d')/(n+1.)
        Q = numpy.interp(subindex, index, Q)
        dQ = numpy.interp(subindex, index, dQ)
    return Q, dQ

class PolarizedQProbe(PolarizedNeutronProbe):
    polarized = True
    def __init__(self, xs=None, name=None, Aguide=270, H=0):
        self._xs = xs
        self._check()
        self.name = name if name is not None else xs[0].name
        self.unique_L = None
        self.Aguide = Parameter.default(Aguide, name="Aguide "+self.name,
                                        limits=[-360, 360])
        self.H = Parameter.default(H, name="H "+self.name)
        self.Q, self.dQ = Qmeasurement_union(xs)
        self.calc_Qo = self.Q

# Deprecated old long name
PolarizedNeutronQProbe = PolarizedQProbe
