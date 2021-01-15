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
import warnings

import numpy as np
from numpy import sqrt, pi, inf, sign, log
import numpy.random
import numpy.fft

from periodictable import nsf, xsf
from bumps.parameter import Parameter, Constant, to_dict
from bumps.plotutil import coordinated_colors, auto_shift
from bumps.data import parse_multi, strip_quotes

from . import fresnel
from .material import Vacuum
from .resolution import QL2T, QT2L, TL2Q, dQdL2dT, dQdT2dLoL, dTdL2dQ
from .resolution import sigma2FWHM, FWHM2sigma, dQ_broadening
from .stitch import stitch
from .reflectivity import convolve, BASE_GUIDE_ANGLE
from .util import asbytes

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
            FWHM angular divergence
        *L* : float or [float] | |Ang|
            Incident wavelength
        *dL* : float or [float] | |Ang|
            FWHM wavelength dispersion
        *data* : ([float], [float])
            R, dR reflectivity measurement and uncertainty
        *dQ* : [float] or None | |1/Ang|
            1-\$sigma$ Q resolution when it cannot be computed directly
            from angular divergence and wavelength dispersion.
        *resolution* : 'normal' or 'uniform'
            Distribution function for Q resolution.

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
           Additional FWHM angular divergence from sample curvature.
           Scale 1-$\sigma$ rms by $2 \surd(2 \ln 2) \approx 2.35$ to convert
           to FWHM.
        *back_reflectivity* : True or False
           True if the beam enters through the substrate

    Measurement properties are fittable parameters.  *theta_offset* in
    particular should be set using *probe.theta_offset.dev(dT)*, with *dT*
    equal to the FWHM uncertainty in the peak position for the rocking curve,
    as measured in radians. Changes to *theta_offset* will then be penalized
    in the cost function for the fit as if it were another measurement.  Use
    :meth:`alignment_uncertainty` to compute dT from the shape of the
    rocking curve.

    Sample broadening adjusts the existing Q resolution rather than
    recalculating it. This allows it the resolution to describe more
    complicated effects than a simple gaussian distribution of wavelength
    and angle will allow. The calculation uses the mean wavelength, angle
    and angular divergence. See :func:`resolution.dQ_broadening` for details.

    *intensity* and *back_absorption* are generally not needed --- scaling
    the reflected signal by an appropriate intensity measurement will correct
    for both of these during reduction.  *background* may be needed,
    particularly for samples with significant hydrogen content due to its
    large isotropic incoherent scattering cross section.

    View properties:

        *view* : string
            One of 'fresnel', 'logfresnel', 'log', 'linear', 'q4', 'residuals'
        *show_resolution* : bool
            True if resolution bars should be plotted with each point.
        *plot_shift* : float
            The number of pixels to shift each new dataset so
            datasets can be seen separately
        *residuals_shift* :
            The number of pixels to shift each new set of residuals
            so the residuals plots can be seen separately.

    Normally *view* is set directly in the class rather than the
    instance since it is not specific to the view.  Fresnel and Q4
    views are corrected for background and intensity; log and
    linear views show the uncorrected data.  The Fresnel reflectivity
    calculation has resolution applied.
    """
    polarized = False
    Aguide = BASE_GUIDE_ANGLE  # default guide field for unpolarized measurements
    view = "log"
    plot_shift = 0
    residuals_shift = 0
    show_resolution = True

    def __init__(self, T=None, dT=0, L=None, dL=0, data=None,
                 intensity=1, background=0, back_absorption=1, theta_offset=0,
                 sample_broadening=0,
                 back_reflectivity=False, name=None, filename=None,
                 dQ=None, resolution='normal'):
        if T is None or L is None:
            raise TypeError("T and L required")
        if sample_broadening is None:
            sample_broadening = 0
        if theta_offset is None:
            theta_offset = 0
        if not name and filename:
            name = os.path.splitext(os.path.basename(filename))[0]
        qualifier = " "+name if name is not None else ""
        self.intensity = Parameter.default(
            intensity, name="intensity"+qualifier)
        self.background = Parameter.default(
            background, name="background"+qualifier, limits=[0., inf])
        self.back_absorption = Parameter.default(
            back_absorption, name="back_absorption"+qualifier, limits=[0., 1.])
        self.theta_offset = Parameter.default(
            theta_offset, name="theta_offset"+qualifier)
        self.sample_broadening = Parameter.default(
            sample_broadening, name="sample_broadening"+qualifier)
        self.back_reflectivity = back_reflectivity
        if data is not None:
            R, dR = data
        else:
            R, dR = None, None

        self._set_TLR(T, dT, L, dL, R, dR, dQ)
        self.name = name
        self.filename = filename
        self.resolution = resolution

    def _set_TLR(self, T, dT, L, dL, R, dR, dQ):
        #if L is None:
        #    L = xsf.xray_wavelength(E)
        #    dL = L * dE/E
        #else:
        #    E = xsf.xray_energy(L)
        #    dE = E * dL/L

        Q = TL2Q(T=T, L=L)
        if dQ is not None:
            dQ = np.asarray(dQ)
        else:
            dQ = dTdL2dQ(T=T, dT=dT, L=L, dL=dL)

        # Make sure that we are dealing with vectors
        T, dT, L, dL = [np.ones_like(Q)*v for v in (T, dT, L, dL)]

        # Probe stores sorted values for convenience of resolution calculator
        idx = np.argsort(Q)
        self.T, self.dT = T[idx], dT[idx]
        self.L, self.dL = L[idx], dL[idx]
        self.Qo, self.dQo = Q[idx], dQ[idx]
        if R is not None:
            R = R[idx]
        if dR is not None:
            dR = dR[idx]
        self.R = R
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
        if self.R is not None:
            self.R = 10**self.R
            if self.dR is not None:
                self.dR = self.R * self.dR * log(10)

    def resynth_data(self):
        """
        Generate new data according to the model R' ~ N(R, dR).

        The resynthesis step is a precursor to refitting the data, as is
        required for certain types of monte carlo error analysis.  The
        first time it is run it will save the original R into Ro.  If you
        reset R in the probe you will also need to reset Ro so that it
        is used for subsequent resynth analysis.
        """
        if not hasattr(self, '_Ro'):
            self._Ro = self.R
        self.R = self._Ro + numpy.random.randn(*self._Ro.shape)*self.dR

    def restore_data(self):
        """
        Restore the original data after resynth.
        """
        self.R = self._Ro
        del self._Ro

    # CRUFT: Ro doesn't need to be part of the public interface.
    @property
    def Ro(self):
        warnings.warn("Use probe.R instead of probe.Ro.", DeprecationWarning)
        return getattr(self, '_Ro', self.R)

    def simulate_data(self, theory, noise=2.):
        r"""
        Set the data for the probe to R + eps with eps ~ normal(dR^2).

        *theory* is (Q, R),

        If the percent *noise* is provided, set dR to R*noise/100 before
        simulating.  *noise* defaults to 2% if no dR is present.

        Note that measured data estimates uncertainty from the number of
        counts.  This means that points above the true value will have
        larger uncertainty than points below the true value.  This bias
        is not captured in the simulated data.
        """
        # Minimum value for dR after noise is added.
        # TODO: does this need to be a parameter?
        noise_floor = 1e-11

        # Set the theory function.
        R = np.array(theory[1], 'd') # Force copy
        assert R.shape == self.Q.shape
        self.R = R

        # Make sure scalar noise is positive.  This check is here to so that
        # old interfaces will fail properly.
        if np.isscalar(noise) and noise <= 0.:
            raise ValueError("Noise level must be positive")

        # If dR is missing then default noise to 2% so that dR will be set.
        if self.dR is None and noise is None:
            noise = 2.

        # Set dR if noise was given or otherwise defaulted.
        if noise is not None:
            self.dR = 0.01 * np.asarray(noise) * self.R
            self.dR[self.dR < noise_floor] = noise_floor

        # Add noise according to dR.
        self.R += numpy.random.randn(*self.R.shape)*self.dR

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
        with open(filename, 'wb') as fid:
            fid.write(asbytes(header))
            data = np.vstack([getattr(self, c) for c in columns])
            np.savetxt(fid, data.T)

    def _set_calc(self, T, L):
        Q = TL2Q(T=T, L=L)

        idx = np.argsort(Q)
        self.calc_T = T[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self.unique_L = np.unique(self.calc_L)
        self._L_idx = np.searchsorted(self.unique_L, L)

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
        if self.sample_broadening.value == 0:
            dQ = self.dQo
        else:
            dQ = dQ_broadening(dQ=self.dQo, L=self.L, T=self.T, dT=self.dT,
                               width=self.sample_broadening.value)
        return dQ

    @dQ.setter
    def dQ(self, dQ):
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
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'filename': self.filename,
            'intensity': self.intensity,
            'background': self.background,
            'back_absorption': self.back_absorption,
            'theta_offset': self.theta_offset,
            'sample_broadening': self.sample_broadening,
        })

    def scattering_factors(self, material, density):
        """
        Returns the scattering factors associated with the material given
        the range of wavelengths/energies used in the probe.
        """
        raise NotImplementedError(
            "need radiation type in <%s> to compute sld for %s"
            % (self.filename, material))

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
        Q = np.arange(self.Qo[0], self.Qo[-1], dQ)
        idx = np.unique(np.searchsorted(self.Qo, Q))
        #print len(idx), len(self.Qo)

        self.T, self.dT = self.T[idx], self.dT[idx]
        self.L, self.dL = self.L[idx], self.dL[idx]
        self.Qo, self.dQo = self.Qo[idx], self.dQo[idx]
        if self.R is not None:
            self.R = self.R[idx]
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

    def Q_c(self, substrate=None, surface=None):
        Srho, Sirho = (0, 0) if substrate is None else substrate.sld(self)[:2]
        Vrho, Virho = (0, 0) if surface is None else surface.sld(self)[:2]
        drho = Srho-Vrho if not self.back_reflectivity else Vrho-Srho
        Q_c = sign(drho)*sqrt(16*pi*abs(drho)*1e-6)
        return Q_c

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

        Note: :meth:`critical_edge` will remove the extra Q calculation
        points introduced by :meth:`oversample`.

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
        Q_c = self.Q_c(substrate, surface)
        Q = np.linspace(Q_c*(1 - delta), Q_c*(1+delta), n)
        L = np.average(self.L)
        T = QL2T(Q=Q, L=L)
        T = np.hstack((self.T, T, 0))
        L = np.hstack((self.L, [L]*(n+1)))
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

        Note: :meth:`oversample` will remove the extra Q calculation
        points introduced by :meth:`critical_edge`.
        """
        if n < 5:
            raise ValueError("Oversampling with n<5 is not useful")

        rng = numpy.random.RandomState(seed=seed)
        T = rng.normal(self.T[:, None], self.dT[:, None], size=(len(self.dT), n-1))
        L = rng.normal(self.L[:, None], self.dL[:, None], size=(len(self.dL), n-1))
        T = np.hstack((self.T, T.flatten()))
        L = np.hstack((self.L, L.flatten()))
        self._set_calc(T, L)

    def _apply_resolution(self, Qin, Rin, interpolation):
        """
        Apply the instrument resolution function
        """
        Q, dQ = _interpolate_Q(self.Q, self.dQ, interpolation)
        if np.iscomplex(Rin).any():
            R_real = convolve(Qin, Rin.real, Q, dQ, resolution=self.resolution)
            R_imag = convolve(Qin, Rin.imag, Q, dQ, resolution=self.resolution)
            R = R_real + 1j*R_imag
        else:
            R = convolve(Qin, Rin, Q, dQ, resolution=self.resolution)
        return Q, R

    def apply_beam(self, calc_Q, calc_R, resolution=True, interpolation=0):
        r"""
        Apply factors such as beam intensity, background, backabsorption,
        resolution to the data.

        *resolution* is True if the resolution function should be applied
        to the reflectivity.

        *interpolation* is the number of Q points to show between the
        nominal Q points of the probe. Use this to draw a smooth theory
        line between the data points. The resolution dQ is interpolated
        between the resolution of the surrounding Q points.

        If an amplitude signal is provided, $r$ will be scaled by
        $\surd I + i \surd B / |r|$, which when squared will equal
        $I |r|^2 + B$. The resolution function will be applied directly
        to the amplitude. Unlike intensity and background, the resulting
        $|G \ast r|^2 \ne G \ast |r|^2$ for convolution operator $\ast$,
        but it should be close.
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
            Q, R = self.Q, np.interp(Q, calc_Q, calc_R)
        if np.iscomplex(R).any():
            # When R is an amplitude you can scale R by sqrt(A) to reproduce
            # the effect of scaling the intensity in the reflectivity. To
            # reproduce the effect of adding a background you can fiddle the
            # phase of r as well using:
            #      s = (sqrt(A) + i sqrt(B)/|r|) r
            # then
            #      |s|^2 = |sqrt(A) + i sqrt(B)/|r||^2 |r|^2
            #            = (A + B/|r|^2) |r|^2
            #            = A |r|^2 + B
            # Note that this cannot work for negative background since
            # |s|^2 >= 0 always, whereas negative background could push the
            # reflectivity below zero.
            R = np.sqrt(self.intensity.value)*R
            if self.background.value > 0:
                R += 1j*np.sqrt(self.background.value)*R/abs(R)
        else:
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
        #I = np.ones_like(self.Q)
        I = 1
        calculator = fresnel.Fresnel(rho=Srho*I, irho=Sirho*I,
                                     Vrho=Vrho*I, Virho=Virho*I)
        return calculator

    def save(self, filename, theory, substrate=None, surface=None):
        """
        Save the data and theory to a file.
        """
        fresnel_calculator = self.fresnel(substrate, surface)
        Q, FQ = self.apply_beam(self.calc_Q, fresnel_calculator(self.calc_Q))
        Q, R = theory
        if len(Q) != len(self.Q):
            # Saving interpolated data
            A = np.array((Q, R, np.interp(Q, self.Q, FQ)))
            header = ("# %17s %20s %20s\n"
                      % ("Q (1/A)", "theory", "fresnel"))
        elif getattr(self, 'R', None) is not None:
            A = np.array((self.Q, self.dQ, self.R, self.dR,
                             R, FQ))
            header = ("# %17s %20s %20s %20s %20s %20s\n"
                      % ("Q (1/A)", "dQ (1/A)", "R", "dR", "theory", "fresnel"))
        else:
            A = np.array((self.Q, self.dQ, R, FQ))
            header = ("# %17s %20s %20s %20s\n"
                      % ("Q (1/A)", "dQ (1/A)", "theory", "fresnel"))

        header = ("# intensity: %.15g\n# background: %.15g\n"
                    % (self.intensity.value, self.background.value)) + header

        with open(filename, "wb") as fid:
            #print("saving", A)
            fid.write(asbytes(header))
            np.savetxt(fid, A.T, fmt="%20.15g")

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
        import matplotlib.pyplot as plt
        plt.plot(self.Q, self.dQ,
                 label=self.label(prefix=label, suffix=suffix))
        plt.xlabel(r'Q ($\AA^{-1}$)')
        plt.ylabel(r'Q resolution ($1-\sigma \AA^{-1}$)')
        plt.title('Measurement resolution')


    def plot_linear(self, **kwargs):
        """
        Plot the data associated with probe.
        """
        import matplotlib.pyplot as plt
        self._plot_pair(ylabel='Reflectivity', **kwargs)
        plt.yscale('linear')

    def plot_log(self, **kwargs):
        """
        Plot the data associated with probe.
        """
        import matplotlib.pyplot as plt
        self._plot_pair(ylabel='Reflectivity', **kwargs)
        plt.yscale('log')

    def plot_logfresnel(self, *args, **kw):
        """
        Plot the log Fresnel-normalized reflectivity associated with the probe.
        """
        import matplotlib.pyplot as plt
        self.plot_fresnel(*args, **kw)
        plt.yscale('log')

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
        def scale(Q, dQ, R, dR, interpolation=0):
            Q, fresnel = self.apply_beam(self.calc_Q, F(self.calc_Q),
                                         interpolation=interpolation)
            return Q, dQ, R/fresnel, dR/fresnel
        if substrate is None:
            name = "air:%s" % surface.name
        elif surface is None or isinstance(surface, Vacuum):
            name = substrate.name
        else:
            name = "%s:%s" % (substrate.name, surface.name)
        self._plot_pair(scale=scale, ylabel='R/(R(%s)' % name, **kwargs)

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
        def scale(Q, dQ, R, dR, interpolation=0):
            #Q4 = np.maximum(1e-8*Q**-4, self.background.value)
            Q4 = 1e-8*Q**-4*self.intensity.value + self.background.value
            return Q, dQ, R/Q4, dR/Q4
        #Q4[Q4==0] = 1
        self._plot_pair(scale=scale, ylabel='R (100 Q)^4', **kwargs)

    def _plot_pair(self, theory=None,
                   scale=lambda Q, dQ, R, dR, interpolation=0: (Q, dQ, R, dR),
                   ylabel="", suffix="", label=None,
                   plot_shift=None, **kwargs):
        import matplotlib.pyplot as plt
        c = coordinated_colors()
        plot_shift = plot_shift if plot_shift is not None else Probe.plot_shift
        trans = auto_shift(plot_shift)
        if hasattr(self, 'R') and self.R is not None:
            Q, dQ, R, dR = scale(self.Q, self.dQ, self.R, self.dR)
            if not self.show_resolution:
                dQ = None
            plt.errorbar(Q, R, yerr=dR, xerr=dQ, capsize=0,
                         fmt='.', color=c['light'], transform=trans,
                         label=self.label(prefix=label,
                                          gloss='data',
                                          suffix=suffix))
        if theory is not None:
            # TODO: completely restructure interpolation handling
            # Interpolation is used to show the theory curve between the
            # data points.  The _calc_Q points used to predict theory at
            # the measured data are used for the interpolated Q points, with
            # the resolution function centered on each interpolated value.
            # The result is that when interpolation != 0, there are more
            # theory points than data points, and we will need to accomodate
            # this when computing normalization curves for Fresnel and Q^4
            # reflectivity.
            # Issues with current implementation:
            # * If the resolution is too tight, there won't be sufficient
            #   support from _calc_Q to compute theory at Q interpolated.
            # * dQ for the interpolated points uses linear interpolation
            #   of dQ between neighbours.  If measurements with tight and
            #   loose resolution are interleaved the result will look very
            #   strange.
            # * There are too many assumptions about interpolation shared
            #   between Experiment and Probe objects.  In particular, the
            #   Fresnel object needs to be computed at the same interpolated
            #   points as the theory function.
            # * If there are large gaps in the data the interpolation will
            #   not fill them in correctly.  Perhaps we should set _Q_plot
            #   and _calc_Q_plot independently from the data?
            # * We sometimes show the theory curve without resolution
            #   applied; this has not been tested with interpolation
            interpolation = kwargs.get('interpolation', 0)
            Q, R = theory
            Q, dQ, R, dR = scale(Q, 0, R, 0, interpolation=interpolation)
            plt.plot(Q, R, '-',
                     color=c['dark'], transform=trans,
                     label=self.label(prefix=label,
                                      gloss='theory',
                                      suffix=suffix))
            #from numpy.fft import fft
            #x, y = Q[1::2], abs(fft(R)[:(len(R)-1)//2])
            #y = y * (R.max()/y.max())
            #plt.plot(x, y, '-')
        plt.xlabel('Q (inv Angstroms)')
        plt.ylabel(ylabel)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        h = plt.legend(fancybox=True, numpoints=1)
        h.get_frame().set_alpha(0.5)

    def plot_residuals(self, theory=None, suffix='', label=None,
                       plot_shift=None, **kwargs):
        import matplotlib.pyplot as plt
        plot_shift = plot_shift if plot_shift is not None else Probe.residuals_shift
        trans = auto_shift(plot_shift)
        if theory is not None and self.R is not None:
            c = coordinated_colors()
            Q, R = theory
            # In case theory curve is evaluated at more/different points...
            R = np.interp(self.Q, Q, R)
            residual = (R - self.R)/self.dR
            plt.plot(self.Q, residual,
                     '.', color=c['light'],
                     transform=trans,
                     label=self.label(prefix=label, suffix=suffix))
        plt.axhline(1, color='black', ls='--', lw=1)
        plt.axhline(0, color='black', lw=1)
        plt.axhline(-1, color='black', ls='--', lw=1)
        plt.xlabel('Q (inv A)')
        plt.ylabel('(theory-data)/error')
        plt.legend(numpoints=1)

    def plot_fft(self, theory=None, suffix='', label=None,
                 substrate=None, surface=None, **kwargs):
        """
        FFT analysis of reflectivity signal.
        """
        raise NotImplementedError
        import matplotlib.pyplot as plt
        c = coordinated_colors()
        if substrate is None and surface is None:
            raise TypeError("FFT reflectivity needs substrate or surface")
        F = self.fresnel(substrate=substrate, surface=surface)
        #Qc = sqrt(16*pi*substrate)
        Qc = 0
        Qmax = max(self.Q)
        T = np.linspace(Qc, Qmax, len(self.Q))
        z = np.linspace(0, 2*pi/Qmax, len(self.Q)//2)
        if hasattr(self, 'R'):
            signal = np.interp(T, self.Q, self.R/F(self.Q))
            A = abs(numpy.fft.fft(signal - np.average(signal)))
            A = A[:len(A)//2]
            plt.plot(z, A, '.-', color=c['light'],
                     label=self.label(prefix=label,
                                      gloss='data',
                                      suffix=suffix))
        if theory is not None:
            Q, R = theory
            signal = np.interp(T, Q, R/F(Q))
            A = abs(numpy.fft.fft(signal-np.average(signal)))
            A = A[:len(A)//2]
            plt.plot(z, A, '-', color=c['dark'],
                     label=self.label(prefix=label,
                                      gloss='theory',
                                      suffix=suffix))
        plt.xlabel('w (A)')
        if substrate is None:
            name = "air:%s" % surface.name
        elif surface is None or isinstance(surface, Vacuum):
            name = substrate.name
        else:
            name = "%s:%s" % (substrate.name, surface.name)
        plt.ylabel('|FFT(R/R(%s))|' % name)

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
        self.R = np.hstack([p.R for p in self.probes])
        self.dR = np.hstack([p.dR for p in self.probes])
        self.name = name if name is not None else self.probes[0].name

        back_refls = [f.back_reflectivity for f in self.probes]
        if all(back_refls) or not any(back_refls):
            self.back_reflectivity = back_refls[0]
        else:
            raise ValueError("Cannot mix front and back reflectivity measurements")

    def parameters(self):
        return [p.parameters() for p in self.probes]
    parameters.__doc__ = Probe.parameters.__doc__

    def to_dict(self):
        """ Return a dictionary representation of the parameters """
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'probes': self.probes,
        })

    def resynth_data(self):
        for p in self.probes: p.resynth_data()
        self.R = np.hstack([p.R for p in self.probes])
    resynth_data.__doc__ = Probe.resynth_data.__doc__

    def restore_data(self):
        for p in self.probes: p.restore_data()
        self.R = np.hstack([p.R for p in self.probes])
    restore_data.__doc__ = Probe.restore_data.__doc__

    def simulate_data(self, theory, noise=2.):
        """
        Simulate data, allowing for noise to be a dR array for each Q point.
        """
        Q, R = theory
        dR = np.asarray(noise)
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
        return np.hstack([p.Q for p in self.probes])

    @property
    def calc_Q(self):
        return np.unique(np.hstack([p.calc_Q for p in self.probes]))

    @property
    def dQ(self):
        return np.hstack([p.dQ for p in self.probes])

    @property
    def unique_L(self):
        return np.unique(np.hstack([p.unique_L for p in self.probes]))

    def oversample(self, **kw):
        for p in self.probes:
            p.oversample(**kw)
    oversample.__doc__ = Probe.oversample.__doc__

    def scattering_factors(self, material, density):
        # TODO: support wavelength dependent systems
        return self.probes[0].scattering_factors(material, density)
        # result = [p.scattering_factors(material, density) for p in self.probes]
        # return [np.hstack(v) for v in zip(*result)]
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def apply_beam(self, calc_Q, calc_R, interpolation=0, **kw):
        result = [p.apply_beam(calc_Q, calc_R, **kw) for p in self.probes]
        Q, R = [np.hstack(v) for v in zip(*result)]
        return Q, R

    def fresnel(self, *args, **kw):
        return self.probes[0].fresnel(*args, **kw)
    fresnel.__doc__ = Probe.fresnel.__doc__

    def save(self, filename, theory, substrate=None, surface=None):
        for i, (p, th) in enumerate(self.parts(theory=theory)):
            p.save(filename+str(i+1), th, substrate=substrate, surface=surface)
    save.__doc__ = Probe.save.__doc__

    def plot(self, theory=None, **kw):
        import matplotlib.pyplot as plt
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
                      back_reflectivity=Po.back_reflectivity,
                      resolution=Po.resolution)


def load4(filename, keysep=":", sep=None, comment="#", name=None,
          intensity=1, background=0, back_absorption=1,
          back_reflectivity=False, Aguide=BASE_GUIDE_ANGLE, H=0,
          theta_offset=None, sample_broadening=None,
          L=None, dL=None, T=None, dT=None, dR=None,
          FWHM=False, radiation=None,
          columns=None, data_range=(None, None),
          resolution='normal',
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

    *L*, *dL* in Angstroms can be used to recover angle and angular resolution
    for monochromatic sources where wavelength is fixed and angle is varying.
    These values can also be stored in the file header as::

        # wavelength: 4.75  # Ang
        # wavelength_resolution: 0.02  # Ang (1-sigma)

    *T*, *dT* in degrees can be used to recover wavelength and wavelength
    dispersion for time of flight sources where angle is fixed and wavelength
    is varying, or you can store them in the header of the file::

        # angle: 2  # degrees
        # angular_resolution: 0.2  # degrees (1-sigma)

    If both angle and wavelength are varying in the data, you can specify
    a separate value for each point, such the following::

        # wavelength: [1, 1.2, 1.5, 2.0, ...]
        # wavelength_resolution: [0.02, 0.02, 0.02, ...]

    *dR* can be used to replace the uncertainty estimate for R in the
    file with $\Delta R = R * \text{dR}$.  This allows files with only
    two columns, *Q* and *R* to be loaded.  Note that points with *dR=0*
    are automatically set to the minimum *dR>0* in the dataset.

    Instead of constants, you can provide function, *dT = lambda T: f(T)*,
    *dL = lambda L: f(L)* or *dR = lambda Q, R, dR: f(Q, R, dR)* for more
    complex relationships (with *dR()* returning 1-$\sigma$ $\Delta R$).

    *sample_broadening* in degrees FWHM adds to the angular_resolution.
    Scale 1-$\sigma$ rms by $2 \surd(2 \ln 2) \approx 2.34$ to convert to FWHM.

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
    Default is 'neutron'

    *columns* is a string giving the column order in the file.  Default
    order is "Q R dR dQ".  Note: include dR and dQ even if the file only
    has two or three columns, but put the missing columns at the end.

    *data_range* indicates which data rows to use.  Arguments are the
    same as the list slice arguments, *(start, stop, step)*.  This follows
    the usual semantics of list slicing, *L[start:stop:step]*, with
    0-origin indices, stop is last plus one and step optional.  Use negative
    numbers to count from the end.  Default is *(None, None)* for the entire
    data set.

    *resolution* is 'normal' (default) or 'uniform'. Use uniform if you
    are merging Q points from a finely stepped energy sensitive measurement.
    """
    entries = parse_multi(filename, keysep=keysep, sep=sep, comment=comment)
    if columns:
        actual = columns.split()
        natural = "Q R dR dQ".split()
        column_order = [actual.index(k) for k in natural]
    else:
        column_order = [0, 1, 2, 3]
    index = slice(*data_range)
    probe_args = dict(
        name=name,
        filename=filename,
        intensity=intensity,
        background=background,
        back_absorption=back_absorption,
        back_reflectivity=back_reflectivity,
        theta_offset=theta_offset,
        sample_broadening=sample_broadening,
        resolution=resolution,
    )
    data_args = dict(
        radiation=radiation,
        FWHM=FWHM,
        T=T, L=L, dT=dT, dL=dL, dR=dR,
        column_order=column_order,
        index=index,
    )
    if len(entries) == 1:
        probe = _data_as_probe(entries[0], probe_args, **data_args)
    else:
        data_by_xs = {strip_quotes(entry[0]["polarization"])
                      : _data_as_probe(entry, probe_args, **data_args)
                      for entry in entries}
        if not set(data_by_xs.keys()) <= set('-- -+ +- ++'.split()):
            raise ValueError("Unknown cross sections in: "
                             + ", ".join(sorted(data_by_xs.keys())))
        xs = [data_by_xs.get(xs, None) for xs in ('--', '-+', '+-', '++')]

        if any(isinstance(d, QProbe) for d in xs if d is not None):
            probe = PolarizedQProbe(xs, Aguide=Aguide, H=H)
        else:
            probe = PolarizedNeutronProbe(xs, Aguide=Aguide, H=H)
    return probe

def _data_as_probe(entry, probe_args, T, L, dT, dL, dR, FWHM, radiation,
                   column_order, index):
    name = probe_args['filename']
    header, data = entry
    if len(data) == 2:
        data_Q, data_R = (data[k][index] for k in column_order[:2])
        data_dR, data_dQ = None, None
        if dR is None:
            raise ValueError("Need dR for two column data in %r" % name)
    elif len(data) == 3:
        data_Q, data_R, data_dR = (data[k][index] for k in column_order[:3])
        data_dQ = None
    else:
        data_Q, data_R, data_dR, data_dQ = (data[k][index] for k in column_order)

    if FWHM and data_dQ is not None: # dQ is already 1-sigma when not FWHM
        data_dQ = FWHM2sigma(data_dQ)

    # Override dR in the file if desired.
    # Make sure the computed dR is positive (otherwise chisq is infinite) by
    # choosing the smallest positive uncertainty to replace the invalid values.
    if dR is not None:
        data_dR = dR(data_Q, data_R, data_dR) if callable(dR) else data_R * dR
        data_dR[data_dR <= 0] = np.min(data_dR[data_dR > 0])

    # support calculation of sld from material based on radiation type
    if radiation is not None:
        data_radiation = radiation
    elif 'radiation' in header:
        data_radiation = json.loads(header['radiation'])
    else:
        # Default to neutron data if radiation not given in head.
        data_radiation = 'neutron'
        #data_radiation = None

    if data_radiation == 'xray':
        make_probe = XrayProbe
    elif data_radiation == 'neutron':
        make_probe = NeutronProbe
    else:
        make_probe = Probe

    # Get T,dT,L,dL from header if it is not provided as an argument
    def fetch_key(key, override):
        # Note: pulls header and index pulled from context
        if override is not None:
            return override
        elif key in header:
            v = json.loads(header[key])
            return np.array(v)[index] if isinstance(v, list) else v
        else:
            return None

    # Get T and L, either from user input or from datafile.
    data_T = fetch_key('angle', T)
    data_L = fetch_key('wavelength', L)

    # If one of T and L is missing, reconstruct it from Q
    if data_T is None and data_L is not None:
        data_T = QL2T(data_Q, data_L)
    if data_L is None and data_T is not None:
        data_L = QT2L(data_Q, data_T)

    # Get dT and dL, either from user input or from datafile.
    data_dL = fetch_key('wavelength_resolution', dL)
    data_dT = fetch_key('angular_resolution', dT)
    #print(header['angular_resolution'], data_dT)

    # Support dT = f(T), dL = f(L)
    if callable(data_dT):
        if data_T is None:
            raise ValueError("Need T to determine dT for %r" % name)
        data_dT = data_dT(data_T)
    if callable(data_dL):
        if data_L is None:
            raise ValueError("Need L to determine dL for %r" % name)
        data_dL = data_dL(data_L)

    # Convert input dT,dL to FWHM if necessary.
    if data_dL is not None and not FWHM:
        data_dL = sigma2FWHM(data_dL)
    if data_dT is not None and not FWHM:
        data_dT = sigma2FWHM(data_dT)

    # If dT or dL is missing, reconstruct it from Q.
    if data_dT is None and not any(v is None for v in (data_L, data_dL, data_dQ)):
        data_dT = dQdL2dT(data_Q, data_dQ, data_L, data_dL)
    if data_dL is None and not any(v is None for v in (data_T, data_dT, data_dQ)):
        data_dLoL = dQdT2dLoL(data_Q, data_dQ, data_T, data_dT)
        data_dL = data_dLoL * data_L

    # Check reconstruction if user provided any of T, L, dT, or dL.
    # Also, sample_offset or sample_broadening.
    offset = probe_args['theta_offset']
    broadening = probe_args['sample_broadening']
    if any(v is not None for v in (T, dT, L, dL, offset, broadening)):
        if data_T is None:
            raise ValueError("Need L to determine T from Q for %r" % name)
        if data_L is None:
            raise ValueError("Need T to determine L from Q for %r" % name)
        if data_dT is None:
            raise ValueError("Need dL to determine dT from dQ for %r" % name)
        if data_dL is None:
            raise ValueError("Need dT to determine dL from dQ for %r" % name)

    # Build the probe, or the Q probe if we don't have angle and wavelength.
    if all(v is not None for v in (data_T, data_L, data_dT, data_dL)):
        probe = make_probe(
            T=data_T, dT=data_dT,
            L=data_L, dL=data_dL,
            data=(data_R, data_dR),
            dQ=data_dQ,
            **probe_args)
    else:
        # QProbe doesn't accept theta_offset or sample_broadening
        qprobe_args = probe_args.copy()
        qprobe_args.pop('theta_offset')
        qprobe_args.pop('sample_broadening')
        probe = QProbe(data_Q, data_dQ, data=(data_R, data_dR), **qprobe_args)

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
                 back_reflectivity=False, resolution='normal'):
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
        self.R = R
        self.dR = dR
        self.unique_L = None
        self.calc_Qo = self.Qo
        self.name = name
        self.filename = filename
        self.resolution = resolution

    def scattering_factors(self, material, density):
        raise NotImplementedError(
            "need radiation type and wavelength in <%s> to compute sld for %s"
            % (self.filename, material))
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def oversample(self, n=20, seed=1):
        if n < 5:
            raise ValueError("Oversampling with n<5 is not useful")
        rng = numpy.random.RandomState(seed=seed)
        extra = rng.normal(self.Q, self.dQ, size=(n-1, len(self.Q)))
        calc_Q = np.hstack((self.Q, extra.flatten()))
        self.calc_Qo = np.sort(calc_Q)
    oversample.__doc__ = Probe.oversample.__doc__

    def critical_edge(self, substrate=None, surface=None,
                      n=51, delta=0.25):
        Q_c = self.Q_c(substrate, surface)
        extra = np.linspace(Q_c*(1 - delta), Q_c*(1+delta), n)
        calc_Q = np.hstack((self.Q, extra, 0))
        self.calc_Qo = np.sort(calc_Q)
    critical_edge.__doc__ = Probe.critical_edge.__doc__

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
    T, dT, L, dL = [np.asarray(v) for v in (T, dT, L, dL)]

    # Convert to Q, dQ
    Q = TL2Q(T, L)
    dQ = dTdL2dQ(T, dT, L, dL)

    # Sort by Q
    idx = np.argsort(Q)
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
    Q, dQ = [np.array(v) for v in zip(*sorted(Qset))]
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
    view = None  # Default to Probe.view when None
    show_resolution = None  # Default to Probe.show_resolution when None
    substrate = surface = None
    polarized = True
    def __init__(self, xs=None, name=None, Aguide=BASE_GUIDE_ANGLE, H=0):
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
        mm, mp, pm, pp = self.xs
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'pp': pp,
            'pm': pm,
            'mp': mp,
            'mm': mm,
            'a_guide': self.Aguide,
            'h': self.H,
        })

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

    def simulate_data(self, theory, noise=2.):
        if noise is None or np.isscalar(noise):
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

        idx = np.argsort(Q)
        self.calc_T = T[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self.unique_L = np.unique(self.calc_L)
        self._L_idx = np.searchsorted(self.unique_L, L)

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
                else (x_data.Q, np.interp(x_data.Q, Qth, x_th))
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

        if view is None:
            view = Probe.view  # Default to Probe.view

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
        elif view.startswith('resid'):
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
        import matplotlib.pyplot as plt
        if self.pp is None or self.mm is None:
            raise TypeError("cannot form spin asymmetry plot without ++ and --")

        plot_shift = plot_shift if plot_shift is not None else Probe.plot_shift
        trans = auto_shift(plot_shift)
        pp, mm = self.pp, self.mm
        c = coordinated_colors()
        if hasattr(pp, 'R') and hasattr(mm, 'R') and pp.R is not None and mm.R is not None:
            Q, SA, dSA = spin_asymmetry(pp.Q, pp.R, pp.dR, mm.Q, mm.R, mm.dR)
            if dSA is not None:
                res = (self.show_resolution if self.show_resolution is not None
                       else Probe.show_resolution)
                dQ = pp.dQ if res else None
                plt.errorbar(Q, SA, yerr=dSA, xerr=dQ, fmt='.', capsize=0,
                             label=pp.label(prefix=label, gloss='data'),
                             transform=trans,
                             color=c['light'])
            else:
                plt.plot(Q, SA, '.',
                         label=pp.label(prefix=label, gloss='data'),
                         transform=trans,
                         color=c['light'])
        if theory is not None:
            mm, mp, pm, pp = theory
            Q, SA, _ = spin_asymmetry(pp[0], pp[1], None, mm[0], mm[1], None)
            plt.plot(Q, SA,
                     label=self.pp.label(prefix=label, gloss='theory'),
                     transform=trans,
                     color=c['dark'])
        plt.xlabel(r'Q (\AA^{-1})')
        plt.ylabel(r'spin asymmetry $(R^{++} -\, R^{--}) / (R^{++} +\, R^{--})$')
        plt.legend(numpoints=1)

    def _xs_plot(self, plotter, theory=None, **kwargs):
        import matplotlib.pyplot as plt
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
    Rm = np.interp(Qp, Qm, Rm)
    v = (Rp-Rm)/(Rp+Rm)
    if dRp is not None:
        dRm = np.interp(Qp, Qm, dRm)
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
        # Extend the Q-range by 1/2 interval on either side
        Q = np.hstack((0.5*(3.*Q[0]-Q[1]), Q, 0.5*(3.*Q[-1]-Q[-2])))
        dQ = np.hstack((0.5*(3.*dQ[0]-dQ[1]), dQ, 0.5*(3.*dQ[-1]-dQ[-2])))
        index = np.arange(0, len(Q), dtype='d')
        subindex = np.arange(0, (n+1)*(len(Q)-1)+1, dtype='d')/(n+1.)
        Q = np.interp(subindex, index, Q)
        dQ = np.interp(subindex, index, dQ)
    return Q, dQ

class PolarizedQProbe(PolarizedNeutronProbe):
    polarized = True
    def __init__(self, xs=None, name=None, Aguide=BASE_GUIDE_ANGLE, H=0):
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
