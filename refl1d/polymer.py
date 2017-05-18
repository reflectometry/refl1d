# -*- coding: utf-8 -*-
# This program is public domain
# Authors Paul Kienzle, Richard Sheridan
r"""
Layer models for polymer systems.

Analytic Self-consistent Field (SCF) Brush profile\ [#Zhulina]_\ [#Karim]_

Analytical Self-consistent Field (SCF) Mushroom Profile\ [#Adamuti-Trache]_

Numerical Self-consistent Field (SCF) End-Tethered Polymer
Profile\ [#Cosgrove]_\ [#deVos]_\ [#Sheridan]_


.. [#Zhulina] Zhulina, EB; Borisov, OV; Pryamitsyn, VA; Birshtein, TM (1991)
    "Coil-Globule Type Transitions in Polymers. 1. Collapse of Layers
    of Grafted Polymer Chains", Macromolecules 24, 140-149.

.. [#Karim] Karim, A; Douglas, JF; Horkay, F; Fetters, LJ; Satija, SK (1996)
    "Comparative swelling of gels and polymer brush layers",
    Physica B 221, 331-336. doi:10.1016/0921-4526(95)00946-9

.. [#Adamuti-Trache] Adamuţi-Trache, M., McMullen, W. E. & Douglas, J. F.
    Segmental concentration profiles of end-tethered polymers with
    excluded-volume and surface interactions. J. Chem. Phys. 105, 4798 (1996).

.. [#Cosgrove] Cosgrove, T., Heath, T., Van Lent, B., Leermakers, F. A. M.,
    & Scheutjens, J. M. H. M. (1987). Configuration of terminally attached
    chains at the solid/solvent interface: self-consistent field theory and
    a Monte Carlo model. Macromolecules, 20(7), 1692–1696.
    doi:10.1021/ma00173a041

.. [#deVos] De Vos, W. M., & Leermakers, F. A. M. (2009). Modeling the
    structure of a polydisperse polymer brush. Polymer, 50(1), 305–316.
    doi:10.1016/j.polymer.2008.10.025

.. [#Sheridan] Sheridan, R. J., Beers, K. L., et. al (2014). Direct observation
    of "surface theta" conditions. [in prep]

..  [#Vincent] Vincent, B., Edwards, J., Emmett, S., & Croot, R. (1988).
    Phase separation in dispersions of weakly-interacting particles in
    solutions of non-adsorbing polymer. Colloids and Surfaces, 31, 267–298.
    doi:10.1016/0166-6622(88)80200-2

"""

from __future__ import division, print_function, unicode_literals

__all__ = ["PolymerBrush", "PolymerMushroom", "EndTetheredPolymer",
           "VolumeProfile", "layer_thickness"]

import inspect

import numpy as np

from bumps.parameter import Parameter
from .model import Layer
from . import util
from time import time

try:
    from collections import OrderedDict
except ImportError:
    class OrderedDict(dict):
        def popitem(self, *args, **kw):
            return dict.popitem(self, *args)

from numpy import real, imag, exp, log, sqrt, pi, hstack, ones_like, fabs

# This is okay to use as long as LAMBDA_ARRAY is symmetric,
# otherwise a slice LAMBDA_ARRAY[::-1] is necessary
from numpy.core.multiarray import correlate as raw_convolve

from numpy.core import add
addred = add.reduce

LAMBDA_1 = 1.0/6.0 #always assume cubic lattice (1/6) for now
LAMBDA_0 = 1.0 - 2.0*LAMBDA_1
LAMBDA_ARRAY = np.array([LAMBDA_1, LAMBDA_0, LAMBDA_1])
MINLAT = 25
SQRT_PI = sqrt(pi)

class PolymerBrush(Layer):
    r"""
    Polymer brushes in a solvent

    :Parameters:
        *thickness*
            the thickness of the solvent layer
        *interface*
            the roughness of the solvent surface
        *polymer*
            the polymer material
        *solvent*
            the solvent material or vacuum
        *base_vf*
            volume fraction (%) of the polymer brush at the interface
        *base*
            the thickness of the brush interface (A)
        *length*
            the length of the brush above the interface (A)
        *power*
            the rate of brush thinning
        *sigma*
            rms brush roughness (A)

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi", density=0.965).

    These parameters combine in the following profile formula:

    .. math::

        V(z) &= \left\{
          \begin{array}{ll}
            V_o                        & \mbox{if } z <= z_o \\
            V_o (1 - ((z-z_o)/L)^2)^p  & \mbox{if } z_o < z < z_o + L \\
            0                          & \mbox{if } z >= z_o + L
          \end{array}
        \right. \\
        V_\sigma(z)
           &= V(z) \star
                 \frac{e^{-\frac{1}{2}(z/\sigma)^2}}{\sqrt{2\pi\sigma^2}} \\
        \rho(z) &= \rho_p V_\sigma(z) + \rho_s (1-V_\sigma(z))

    where $V_\sigma(z)$ is volume fraction convoluted with brush
    roughness $\sigma$ and $\rho(z)$ is the complex scattering
    length density of the profile.
    """
    def __init__(self, thickness=0, interface=0, name="brush",
                 polymer=None, solvent=None, base_vf=None,
                 base=None, length=None, power=None, sigma=None):
        prefix = name + " "
        self.thickness = Parameter.default(thickness, name=prefix+"thickness")
        self.interface = Parameter.default(interface, name=prefix+"interface")
        self.base_vf = Parameter.default(base_vf, name=prefix+"base_vf")
        self.base = Parameter.default(base, name=prefix+"base")
        self.length = Parameter.default(length, name=prefix+"length")
        self.power = Parameter.default(power, name=prefix+"power")
        self.sigma = Parameter.default(sigma, name=prefix+"sigma")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name
        # Constraints:
        #   base_vf in [0, 1]
        #   base, length, sigma, thickness, interface>0
        #   base+length+3*sigma <= thickness

    def parameters(self):
        return {'solvent':self.solvent.parameters(),
                'polymer':self.polymer.parameters(),
                'base_vf':self.base_vf,
                'base':self.base,
                'length':self.length,
                'power':self.power,
                'sigma':self.sigma,
               }

    def profile(self, z):
        base_vf, base, length, power, sigma \
            = [p.value for p in (self.base_vf, self.base, self.length,
                                 self.power, self.sigma)]
        base_vf /= 100. # % to fraction
        L0 = base  # if base < thickness else thickness
        L1 = base+length # if base+length < thickness else thickness-L0
        if length == 0:
            v = np.ones_like(z)
        else:
            v = (1 - ((z-L0)/(L1-L0))**2)
        v[z < L0] = 1
        v[z > L1] = 0
        brush_profile = base_vf * v**power
        # TODO: we could use Nevot-Croce rather than smearing the profile
        vf = smear(z, brush_profile, sigma)
        return vf

    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw, Pz = slabs.microslabs(thickness)
        # Skip layer if it falls to zero thickness.  This may lead to
        # problems in the fitter, since R(thickness) is non-differentiable
        # at thickness = 0.  "Clip to boundary" range handling will at
        # least allow this point to be found.
        # TODO: consider using this behaviour on all layer types.
        if len(Pw) == 0:
            return

        Mr, Mi = self.polymer.sld(probe)
        Sr, Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try:
            M, S = M[0], S[0]  # Temporary hack
        except Exception:
            pass

        vf = self.profile(Pz)
        Pw, vf = util.merge_ends(Pw, vf, tol=1e-3)
        P = M*vf + S*(1-vf)
        Pr, Pi = real(P), imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

def layer_thickness(z):
    """
    Return the thickness of a layer given the microslab z points.

    The z points are at the centers of the bins.  we can use the recurrence
    that boundary b[k] = z[k-1] + (z[k-1] - b[k-1]) to compute the
    total length of the layer.
    """
    return 2 * (np.sum(z[-1::-2]) - np.sum(z[-2::-2]))

class VolumeProfile(Layer):
    """
    Generic volume profile function

    :Parameters:

        *thickness*
            the thickness of the solvent layer
        *interface*
            the roughness of the solvent surface
        *material*
            the polymer material
        *solvent*
            the solvent material
        *profile*
            the profile function, suitably parameterized

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi", density=0.965).

    These parameters combine in the following profile formula::

        sld = material.sld * profile + solvent.sld * (1 - profile)

    The profile function takes a depth z and returns a density rho.

    For volume profiles, the returned rho should be the volume fraction
    of the material.  For SLD profiles, rho should be complex scattering
    length density of the material.

    Fitting parameters are the available named arguments to the function.
    The first argument must be *z*, which is the array of depths at which
    the profile is to be evaluated.  It is guaranteed to be increasing, with
    step size 2*z[0].

    Initial values for the function parameters can be given using name=value.
    These values can be scalars or fitting parameters.  The function will
    be called with the current parameter values as arguments.  The layer
    thickness can be computed as :func: `layer_thickness`.

    """
    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(self, thickness=0, interface=0, name="VolumeProfile",
                 material=None, solvent=None, profile=None, **kw):
        if interface != 0:
            raise NotImplementedError("interface not yet supported")
        if profile is None or material is None or solvent is None:
            raise TypeError("Need polymer, solvent and profile")
        self.name = name
        self.thickness = Parameter.default(thickness, name="solvent thickness")
        self.interface = Parameter.default(interface, name="solvent interface")
        self.profile = profile
        self.solvent = solvent
        self.material = material

        # Query profile function for the list of arguments
        vars = inspect.getargspec(profile)[0]
        #print("vars", vars)
        if inspect.ismethod(profile):
            vars = vars[1:]  # Chop self
        vars = vars[1:]  # Chop z
        #print(vars)
        unused = [k for k in kw.keys() if k not in vars]
        if len(unused) > 0:
            raise TypeError("Profile got unexpected keyword argument '%s'"%unused[0])
        dups = [k for k in vars
                if k in ('thickness', 'interface', 'polymer', 'solvent', 'profile')]
        if len(dups) > 0:
            raise TypeError("Profile has conflicting argument '%s'"%dups[0])
        for k in vars:
            kw.setdefault(k, 0)
        for k, v in kw.items():
            setattr(self, k, Parameter.default(v, name=k))

        self._parameters = vars

    def parameters(self):
        P = {'solvent':self.solvent.parameters(),
             'material':self.material.parameters(),
            }
        for k in self._parameters:
            P[k] = getattr(self, k)
        return P

    def render(self, probe, slabs):
        Mr, Mi = self.material.sld(probe)
        Sr, Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        #M, S = M[0], S[0]  # Temporary hack
        Pw, Pz = slabs.microslabs(self.thickness.value)
        if len(Pw) == 0:
            return
        kw = dict((k, getattr(self, k).value) for k in self._parameters)
        #print(kw)
        phi = self.profile(Pz, **kw)
        try:
            if phi.shape != Pz.shape:
                raise Exception
        except Exception:
            raise TypeError("profile function '%s' did not return array phi(z)"
                            % self.profile.__name__)
        Pw, phi = util.merge_ends(Pw, phi, tol=1e-3)
        P = M*phi + S*(1-phi)
        slabs.extend(rho=[real(P)], irho=[imag(P)], w=Pw)
        #slabs.interface(self.interface.value)


def smear(z, P, sigma):
    """
    Gaussian smearing

    :Parameters:
        *z* | vector
            equally spaced sample times
        *P* | vector
            sample values
        *sigma* | real
            root-mean-squared convolution width
    :Returns:
        *Ps* | vector
            smeared sample values
    """
    if len(z) < 3:
        return P
    dz = z[1]-z[0]
    if 3*sigma < dz:
        return P
    w = int(3*sigma/dz)
    G = exp(-0.5*(np.arange(-w, w+1)*(dz/sigma))**2)
    full = np.hstack(([P[0]]*w, P, [P[-1]]*w))
    return np.convolve(full, G/np.sum(G), 'valid')

class PolymerMushroom(Layer):
    r"""
    Polymer mushrooms in a solvent (volume profile)

    :Parameters:
        *delta* | real scalar
            interaction parameter
        *vf* | real scalar
            not quite volume fraction (dimensionless grafting density)
        *sigma* | real scalar
            convolution roughness (A)

    Using analytical SCF methods for gaussian chains, which are scaled
    by the radius of gyration of the equivalent free polymer as an
    approximation to results of renormalization group methods.\ [#Adamuti-Trache]_

    Solutions are only strictly valid for vf << 1.
    """

    def __init__(self, thickness=0, interface=0, name="Mushroom",
                 polymer=None, solvent=None, sigma=0,
                 vf=0, delta=0):
        self.thickness = Parameter.default(thickness, name="Mushroom thickness")
        self.interface = Parameter.default(interface, name="Mushroom interface")
        self.delta = Parameter.default(delta, name="delta")
        self.vf = Parameter.default(vf, name="vf")
        self.sigma = Parameter.default(sigma, name="sigma")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name

    def parameters(self):
        return {'solvent':self.solvent.parameters(),
                'polymer':self.polymer.parameters(),
                'delta':self.delta,
                'vf':self.vf,
                'sigma':self.sigma,
                'thickness':self.thickness,
                'interface':self.interface
               }

    def profile(self, z):
        delta, sigma, vf, thickness \
            = [p.value for p in (self.delta, self.sigma, self.vf, self.thickness)]

        return smear(z, MushroomProfile(z, delta, vf, sigma), sigma)


    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw, Pz = slabs.microslabs(thickness)
        # Skip layer if it falls to zero thickness.  This may lead to
        # problems in the fitter, since R(thickness) is non-differentiable
        # at thickness = 0.  "Clip to boundary" range handling will at
        # least allow this point to be found.
        # TODO: consider using this behaviour on all layer types.
        if len(Pw) == 0:
            return

        Mr, Mi = self.polymer.sld(probe)
        Sr, Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try:
            M, S = M[0], S[0]  # Temporary hack
        except KeyError:
            pass

        phi = self.profile(Pz)
        Pw, phi = util.merge_ends(Pw, phi, tol=1e-3)
        P = M*phi + S*(1-phi)
        Pr, Pi = np.real(P), np.imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

def MushroomProfile(z, delta=0.1, vf=1.0, sigma=1.0):
    thickness = layer_thickness(z)
    thresh = 1e-10

    base = 3.0*sigma # tail is erf, capture 95% of the mixing
    Rg = (thickness-base) / 4.0 # profile ends by ~4 RG, so we can tether these
    keep = (z-base) >= 0.0
    x = (z[keep] - base) / Rg

    """
    mushroom_profile_math has a divide by zero problem at delta=0.
    Fix it by weighted average of the profile above and below a threshold.
    No visual difference when delta is between +-0.001, and there's no
    floating point error until ~+-1e-14.
    """

    if fabs(delta) > thresh:
        mushroom_profile = mushroom_math(x, delta, vf)
    else: # we should RARELY get here
        scale = (delta+thresh)/2.0/thresh
        mushroom_profile = (scale*mushroom_math(x, thresh, vf)
                            + (1.0-scale)*mushroom_math(x, -thresh, vf))

    try:
        # make the base connect with the profile
        zextra = z[np.logical_not(keep)]
        base_profile = ones_like(zextra)*mushroom_profile[0]
    except IndexError:
        base_profile = ones_like(z)*mushroom_profile[0]

    return hstack((base_profile, mushroom_profile))

def mushroom_math(x, delta=.1, vf=.1):
    """
    new method, rewrite for numerical stability at high delta
    delta=0 causes divide by zero error!! Compensate elsewhere.
    http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
    """

    from scipy.special import erfc, erfcx

    x_half = x/2.0
    delta_double = 2.0*delta
    return (
            (
             erfc(x_half)
             - erfcx(delta_double+x_half)/exp(x_half*x_half)
             - erfc(x)
             + (
                (.25-delta*(x+delta_double))*erfcx(delta_double+x)
                + delta/SQRT_PI
               ) * 4.0 / exp(x*x)
            ) * vf / (delta_double * erfcx(delta_double))
           )

class EndTetheredPolymer(Layer):
    r"""
    Polymer end-tethered to an interface in a solvent

    Uses a numerical self-consistent field profile.\ [#Cosgrove]_\ [#deVos]_\ [#Sheridan]_

    **Parameters**

        *chi*
            solvent interaction parameter
        *chi_s*
            surface interaction parameter
        *h_dry*
            thickness of the neat polymer layer
        *l_lat*
            real length per lattice site
        *mn*
            Number average molecular weight
        *m_lat*
            real mass per lattice segment
        *pdi*
            Dispersity (Polydispersity index)
        *thickness*
            Slab thickness should be greater than the contour
            length of the polymer
        *interface*
            should be zero
        *material*
            the polymer material
        *solvent*
            the solvent material

    Previous layer should not have roughness! Use a spline to simulate it.

    According to [#Vincent]_, $l_\text{lat}$ and $m_\text{lat}$ should be
    calculated by the formulas:

    .. math::

        l_\text{lat} &= \frac{a^2 m/l}{p_l}

        m_\text{lat} &= \frac{(a m/l)^2}{p_l}

    where $l$ is the real polymer's bond length, $m$ is the real segment mass,
    and $a$ is the ratio between molecular weight and radius of gyration at
    theta conditions. The lattice persistence, $p_l$, is:

    .. math::

        p_l = \frac16 \frac{1+1/Z}{1-1/Z}

    with coordination number $Z = 6$ for a cubic lattice, $p_l = .233$.
    """

    def __init__(self, thickness=0, interface=0, name="EndTetheredPolymer",
                 polymer=None, solvent=None, chi=0, chi_s=0, h_dry=None,
                 l_lat=1, mn=None, m_lat=1, pdi=1):
        if interface != 0:
            raise NotImplementedError("interface not yet supported")
        if polymer is None or solvent is None or h_dry is None or mn is None:
            raise TypeError("Need polymer, solvent and profile")

        self.thickness = Parameter.default(thickness, name="SCF thickness")
        self.interface = Parameter.default(interface, name="SCF interface")
        self.chi = Parameter.default(chi, name="chi")
        self.chi_s = Parameter.default(chi_s, name="surface chi")
        self.h_dry = Parameter.default(h_dry, name="dry thickness")
        self.l_lat = Parameter.default(l_lat, name="lattice layer length")
        self.mn = Parameter.default(mn, name="Num. Avg. MW")
        self.m_lat = Parameter.default(m_lat, name="lattice segment mass")
        self.pdi = Parameter.default(pdi, name="Dispersity")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name

    def parameters(self):
        return {'solvent':self.solvent.parameters(),
                'polymer':self.polymer.parameters(),
                'chi':self.chi,
                'chi_s':self.chi_s,
                'h_dry':self.h_dry,
                'l_lat':self.l_lat,
                'mn':self.mn,
                'm_lat':self.m_lat,
                'pdi':self.pdi,
                'thickness':self.thickness,
                'interface':self.interface
               }

    def profile(self, z):
        return SCFprofile(z, chi=self.chi.value, chi_s=self.chi_s.value,
                          h_dry=self.h_dry.value, l_lat=self.l_lat.value,
                          mn=self.mn.value, m_lat=self.m_lat.value,
                          pdi=self.pdi.value)

    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw, Pz = slabs.microslabs(thickness)
        # Skip layer if it falls to zero thickness.  This may lead to
        # problems in the fitter, since R(thickness) is non-differentiable
        # at thickness = 0.  "Clip to boundary" range handling will at
        # least allow this point to be found.
        # TODO: consider using this behaviour on all layer types.
        if len(Pw) == 0:
            return

        Mr, Mi = self.polymer.sld(probe)
        Sr, Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try:
            M, S = M[0], S[0]  # Temporary hack
        except KeyError:
            pass

        phi = self.profile(Pz)
        Pw, phi = util.merge_ends(Pw, phi, tol=1e-3)
        P = M*phi + S*(1-phi)
        Pr, Pi = np.real(P), np.imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

def SCFprofile(z, chi=None, chi_s=None, h_dry=None, l_lat=1, mn=None,
               m_lat=1, pdi=1, disp=False):
    """
    Generate volume fraction profile for Refl1D based on real parameters.

    The field theory is a lattice-based one, so we need to move between lattice
    and real space. This is done using the parameters l_lat and m_lat, the
    lattice size and the mass of a lattice segment, respectivley. We use h_dry
    (dry thickness) as a convenient measure of surface coverage, along with mn
    (number average molecular weight) as the real inputs.

    Make sure your inputs for h_dry/l_lat and mn/m_lat match dimensions!
    Angstroms and daltons are good choices.

    This function is suitable for use as a VolumeProfile, as well as the
    default EndTetheredPolymer class.
    """

    # calculate lattice space parameters
    theta = h_dry/l_lat
    segments = mn/m_lat
    sigma = theta/segments

    # solve the self consistent field equations using the cache
    if disp:
        print("\n=====Begin calculations=====\n")
    phi_lat = SCFcache(chi, chi_s, pdi, sigma, segments, disp)
    if disp:
        print("\n============================\n")

    # re-dimensionalize the solution
    layers = len(phi_lat)
    z_end = l_lat*layers
    z_lat = np.linspace(0.0, z_end, num=layers)
    phi = np.interp(z, z_lat, phi_lat, right=0.0)

    return phi


def SCFcache(chi, chi_s, pdi, sigma, segments, disp=False, cache=OrderedDict()):
    """
    Return a memoized SCF result by walking from a previous solution.

    Using an OrderedDict (because I want to prune keys FIFO)
    """
    # prime the cache with a known easy solution
    if not cache:
        cache[(0, 0, 0, .1, .2)] = SCFsolve(sigma=.1, segments=100)

    if disp:
        starttime = time()

    # Try to keep the parameters between 0 and 1. Factors are arbitrary.
    scaled_parameters = (chi, chi_s*3, pdi-1, sigma, segments/500)

    # longshot, but return a cached result if we hit it
    if scaled_parameters in cache:
        if disp:
            print('SCFcache hit at:', scaled_parameters)
        phi = cache.pop(scaled_parameters) # pop and assign to shift the key
        cache[scaled_parameters] = phi     # to the end as "recently used"
        return phi

    # Find the closest parameters in the cache: O(len(cache))

    # Numpy setup
    cached_parameters = list(cache)
    cp_array = np.array(cached_parameters)
    p_array = np.array(scaled_parameters)

    # Calculate distances to all cached parameters
    deltas = p_array - cp_array # Parameter space displacement vectors
    norms = sqrt(addred(deltas*deltas, axis=1)) # and their magnitudes
    closest_index = norms.argmin()

    # Organize closest point data for later use
    closest_cp = cached_parameters[closest_index]
    closest_cp_array = cp_array[closest_index]
    closest_delta = deltas[closest_index]

    phi0 = cache.pop(closest_cp) # pop and assign to shift the key
    cache[closest_cp] = phi0     # to the end as "recently used"

    if disp:
        print("Walking from nearest:", closest_cp_array)
        print("to:", p_array)

    """
    We must walk from the previously cached point to the desired region.
    This is goes from step=0 (cached) and step=1 (finish), where the step=0
    is implicit above. We try the full step first, so that this function only
    calls SCFsolve one time during normal cache misses.

    The solver may not converge if the step size is too big. In that case,
    we retry with half the step size. This should find the edge of the basin
    of attraction for the solution eventually. On successful steps we increase
    stepsize slightly to accelerate after getting stuck.

    It might seem to make sense to bin parameters into a coarser grid, so we
    would be more likely to have cache hits and use them, but this rarely
    happened in practice.
    """

    step = 1.0 # Fractional distance between cached and requested
    dstep = 1.0 # Step size increment
    flag = True

    while flag:
        # end on 1.0 exactly every time
        if step >= 1.0:
            step = 1.0
            flag = False

        # conditional math because, "why risk floating point error"
        if flag:
            p_tup = tuple(closest_cp_array + step*closest_delta)
        else:
            p_tup = scaled_parameters

        if disp:
            print('Parameter step is', step)
            print('current parameters:', p_tup)

        parameters = (p_tup[0], p_tup[1]/3, p_tup[2]+1,
                      p_tup[3], p_tup[4]*500)
        try:
            phi0 = SCFsolve(*parameters, phi0=phi0, disp=disp)
            cache[p_tup] = phi0
            dstep *= 1.05
            step += dstep
        except RuntimeError as e:
            if hasattr(e, 'message'):
                message = e.message
            elif hasattr(e, 'args'):
                message = e.args[0]
            else:
                raise

            if message != "solver couldn't converge":
                raise
            else:
                flag = True # Reset this so we don't quit if step=1.0 fails
                dstep *= .5
                step -= dstep

    if disp:
        print('SCFcache execution time:', round(time()-starttime, 3), "s")

    # keep the cache from consuming all things
    if len(cache) > 1000:
        if disp:
            print('pruning cache')
        for i in range(100):
            cache.popitem(last=False)

    return phi0


def SCFsolve(chi=0, chi_s=0, pdi=1, sigma=None, segments=None,
             disp=False, phi0=None, maxiter=15):
    """
    Solve SCF equations using an initial guess and lattice parameters

    This function finds a solution for the equations where the lattice size
    is sufficiently large.

    The Newton-Krylov solver really makes this one. Krylov+gmres was faster
    than the other scipy.optimize alternatives by quite a lot.
    """

    from scipy.optimize import root

    if sigma >= 1:
        raise ValueError('Chains that short cannot be squeezed that high')

    if disp:
        starttime = time()

    p_i = SZdist(pdi, segments)

    if phi0 is None:
        # TODO: Better initial guess for chi>.6
        layers, phi0 = default_guess(segments, sigma)
        if disp:
            print('No guess passed, using default phi0: layers =', layers)
    else:
        phi0 = fabs(phi0)
        phi0[phi0 > .99999] = .99999
        layers = len(phi0)
        if disp:
            print("Initial guess passed: layers =", layers)

    # Loop resizing variables

    # We tolerate up to 2ppm of our polymer in the last layer,
    theta = sigma*segments
    tol = 2e-6*theta
    # otherwise we grow it by 20%.
    ratio = .2

    # callback to detect an undersized lattice early
    def callback(x, fx):
        short_circuit_callback(x, tol)

    # other loop variables
    jac_solve_method = 'gmres'

    while True:
        if disp:
            print("Solving SCF equations")

        try:
            result = root(
                SCFeqns, phi0, args=(chi, chi_s, sigma, segments, p_i),
                method='Krylov', callback=callback,
                options={'disp':bool(disp), 'maxiter':maxiter,
                         'jac_options':{'method':jac_solve_method}})
            if disp:
                print('Solver exit code:', result.status, result.message)

            if result.status == 1:
                # success! carry on to resize logic.
                phi = fabs(result.x)
            elif result.status == 2:
                raise RuntimeError("solver couldn't converge")

        except ShortCircuitError as e:
            # dumping out to resize since we've exceeded resize tol by 4x
            phi = fabs(e.x)
            if disp:
                print(e.value)

        except ValueError as e:
            if hasattr(e, 'message'):
                message = e.message
            elif hasattr(e, 'args'):
                message = e.args[0]
            else:
                raise

            if message == 'array must not contain infs or NaNs':
                # TODO: Handle this error better. Caused by double overflows.
                raise #RuntimeError("solver couldn't converge")
            else:
                raise

        except RuntimeError as e:
            if hasattr(e, 'message'):
                message = e.message
            elif hasattr(e, 'args'):
                message = e.args[0]
            else:
                raise
            if message == 'gmres is not re-entrant':
                # Threads are racing to use gmres. Lose the race and use
                # something slower but thread-safe.
                jac_solve_method = 'lgmres'
                continue
            else:
                raise

        if disp:
            print('phi(M)/sum(phi) =', phi[-1] / theta * 1e6, '(ppm)')

        if phi[-1] > tol:
            # if the last layer is beyond tolerance, grow the lattice
            newlayers = max(1, int(round(len(phi0)*ratio)))
            if disp:
                print('Growing undersized lattice by', newlayers)
            phi0 = hstack((phi, np.linspace(phi[-1], 0, num=newlayers)))
        else:
            # otherwise, we are done for real
            break

    # chop off extra layers
    chop = addred(phi > tol) + 1
    phi = phi[:max(MINLAT, chop)]

    if disp:
        print('After chopping: phi(M)/sum(phi) =',
              phi[-1] / theta * 1e6, '(ppm)')
        print("lattice size:", len(phi))
        print("SCFsolve execution time:", round(time()-starttime, 3), "s")

    return phi

def SZdist(pdi, nn, cache=OrderedDict()):
    """
    Calculate Shultz-Zimm distribution from PDI and number average DP

    Shultz-Zimm is a "realistic" distribution for linear polymers. Numerical
    problems arise when the distribution gets too uniform, so if we find them,
    default to an exact uniform calculation.
    """
    args = pdi, nn
    if args in cache:
        p_ni = cache.pop(args)
        cache[args] = p_ni
        return p_ni

    from scipy.special import gammaln

    uniform = False

    if pdi == 1.0:
        uniform = True
    elif pdi < 1.0:
        raise ValueError('Invalid PDI')
    else:
        x = 1.0/(pdi-1.0)
        # Calculate the distribution in chunks so we don't waste CPU time
        chunk = 256
        p_ni_list = []
        pdi_underflow = False

        for i in range(max(1, int((100*nn)/chunk))):
            ni = np.arange(chunk*i+1, chunk*(i+1)+1, dtype=np.float64)
            r = ni/nn
            xr = x*r

            p_ni = exp(log(x/ni) - gammaln(x+1) + xr*(log(xr)/r-1))

            pdi_underflow = (p_ni >= 1.0).any() # catch "too small PDI"
            if pdi_underflow:
                break # and break out to uniform calculation

            # Stop calculating when species account for less than 1ppm
            keep = (r < 1.0) | (p_ni >= 1e-6)
            if keep.all():
                p_ni_list.append(p_ni)
            else:
                p_ni_list.append(p_ni[keep])
                break
        else: # Belongs to the for loop. Executes if no break statement runs.
            raise RuntimeError('SZdist overflow')

    if uniform or pdi_underflow:
        # NOTE: rounding here allows nn to be a double in the rest of the logic
        p_ni = np.zeros((1, int(round(nn))))
        p_ni[0, -1] = 1.0
    else:
        p_ni = hstack(p_ni_list).reshape(1, -1)

    cache[args] = p_ni

    if len(cache) > 9000:
        for i in range(1000):
            cache.popitem(last=False)

    return p_ni

def default_guess(segments=100, sigma=.5, chi=0, chi_s=0):
    """
    Produce an initial guess for phi via analytical approximants.

    For now, a line using numbers from scaling theory
    """
    ss = sqrt(sigma)
    default_layers = int(round(max(MINLAT, segments*ss)))
    default_phi0 = np.linspace(ss, 0, num=default_layers)
    return default_layers, default_phi0

class ShortCircuitError(Exception):
    """
    Special error to stop root() before a solution is found.
    """
    def __init__(self, value, x):
        self.value = value
        self.x = x

    def __str__(self):
        return repr(self.value)

def short_circuit_callback(x, tol):
    """
    Special callback to stop root() before solution is found.

    This kills root if the tolerances are exceeded by 4 times the tolerances
    of the lattice resizing loop. This seems to work well empirically to
    restart the solver when necessary without cutting out of otherwise
    reasonable solver trajectories.
    """
    if abs(x[-1]) > 4*tol:
        raise ShortCircuitError('Stopping, lattice too small!', x)

def SCFeqns(phi_z, chi, chi_s, sigma, navgsegments, p_i):
    """
    System of SCF equation for terminally attached polymers.

    Formatted for input to a nonlinear minimizer or solver.
    """

    # let the solver go negative if it wants
    phi_z = fabs(phi_z)

    # attempts to try fields with values greater than one are penalized
    toomuch = phi_z > .99999
    if toomuch.any():
        penalty = np.where(toomuch, 1e5*(phi_z-.99999), 0)
        phi_z[toomuch] = .99999
    else:
        penalty = 0.0

    layers = phi_z.size
    cutoff = p_i.size

    # calculate all needed quantities for new g_z
    delta = np.zeros(layers)
    delta[0] = 1.0
    phi_z_avg = calc_phi_z_avg(phi_z)

    # calculate new g_z (Boltzmann weighting factors)
    g_z = (1.0 - phi_z)*exp(2*chi*phi_z_avg + delta*chi_s)

    # normalize g_z for numerical stability
    u = -log(g_z)
    uavg = addred(u)/layers
    g_z_norm = g_z*exp(uavg)

    # calculate weighting factors for terminally attached chains
    g_zs_ta_norm = calc_g_zs(g_z_norm, 0, layers, cutoff)

    # calculate normalization constants from 1/(single chain partition fn)
    if cutoff == round(navgsegments): # if uniform,
        c_i_norm = sigma/addred(g_zs_ta_norm[:, -1]) # take a shortcut!
    else:
        c_i_norm = sigma*p_i/addred(g_zs_ta_norm, axis=0)

    # calculate weighting factors for free chains
    g_zs_free_ngts_norm = calc_g_zs(g_z_norm, c_i_norm, layers, cutoff)

    # calculate new polymer density field
    phi_z_new = calc_phi_z(g_zs_ta_norm, g_zs_free_ngts_norm, g_z_norm)

    # Handle float overflows only if they show themselves
    if np.isnan(phi_z_new).any():
        maxfloat = _getmax(g_zs_ta_norm.dtype.type)
        g_zs_ta_norm[np.isinf(g_zs_ta_norm)] = maxfloat
        g_zs_free_ngts_norm[np.isinf(g_zs_free_ngts_norm)] = maxfloat
        phi_z_new = calc_phi_z(g_zs_ta_norm, g_zs_free_ngts_norm, g_z_norm)

    eps_z = phi_z - phi_z_new
    return eps_z + penalty*np.sign(eps_z)

def _getmax(t, seen_t={}):
    try:
        return seen_t[t]
    except KeyError:
        from numpy.core import getlimits
        fmax = getlimits.finfo(t).max
        seen_t[t] = fmax
        return fmax

def calc_phi_z_avg(phi_z):
    return raw_convolve(phi_z, LAMBDA_ARRAY, 1)

def calc_phi_z(g_ta, g_free, g_z):
    return addred(g_ta*np.fliplr(g_free), axis=1)/g_z

def calc_g_zs(g_z, c_i, layers, segments):
    # initialize
    g_zs = np.empty((layers, segments), dtype=np.float64, order='F')

    # choose special case
    if np.size(c_i) == 1:
        if c_i:
            # uniform chains
            g_zs[:, 0] = c_i*g_z
        else:
            # terminally attached ends
            g_zs[:, 0] = 0.0
            g_zs[0, 0] = g_z[0]
        from refl1d.calc_g_zs_cex import _calc_g_zs_uniform
        _calc_g_zs_uniform(g_z, g_zs, LAMBDA_0, LAMBDA_1, layers, segments)

    else:
        # free ends
        g_zs[:, 0] = c_i[0, -1]*g_z
        from refl1d.calc_g_zs_cex import _calc_g_zs
        _calc_g_zs(g_z, c_i, g_zs, LAMBDA_0, LAMBDA_1, layers, segments)

    # Older versions of inner loops

#    if np.size(c_i)==1:
#        c_i = np.zeros((1, int(round(segments))))
#        g_zs[:, 0] = 0.0
#        g_zs[0, 0] = g_z[0]
#    else:
#        # free chains
#        g_zs[:, 0] = c_i[0, segments-1]*g_z

    # FASTEST: call some custom C code identical to "SLOW" loop
#    _calc_g_zs_pointers(g_z, c_i, g_zs, LAMBDA_0, LAMBDA_1, layers, segments)

    # FASTER: use the convolve function to partially vectorize
#    pg_zs=g_zs[:, 0]
#    for r in range(1, segments):
#        pg_zs=g_z*(c_i[0, segments-r-1]+raw_convolve(pg_zs, LAMBDA_ARRAY, 1))
#        g_zs[:, r]=pg_zs

    # SLOW: loop outright, pulling some slicing out of the innermost loop
#    for r in range(1, segments):
#        c=c_i[0, segments-r-1]
#        g_zs[0, r]=(pg_zs[0]*LAMBDA_0+pg_zs[1]*LAMBDA_1+c)*g_z[0]
#        for z in range(1, (layers-1)):
#            g_zs[z, r]=(pg_zs[z-1]*LAMBDA_1
#                       + pg_zs[z]*LAMBDA_0
#                       + pg_zs[z+1]*LAMBDA_1
#                       + c) * g_z[z]
#        g_zs[-1, r]=(pg_zs[-1]*LAMBDA_0+pg_zs[-2]*LAMBDA_1+c)*g_z[-1]
#        pg_zs=g_zs[:, r]

    return g_zs
