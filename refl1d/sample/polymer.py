# -*- coding: utf-8 -*-
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

.. [#Sheridan] Sheridan, R. J., Orski, S. V., Jones, R. L., Satija, S.,
    & Beers, K. L. (2017). Surface interaction parameter measurement of
    solvated polymers via model end-tethered chains. [Submitted]

..  [#Vincent] Vincent, B., Edwards, J., Emmett, S., & Croot, R. (1988).
    Phase separation in dispersions of weakly-interacting particles in
    solutions of non-adsorbing polymer. Colloids and Surfaces, 31, 267–298.
    doi:10.1016/0166-6622(88)80200-2

"""

__all__ = ["PolymerBrush", "PolymerMushroom", "EndTetheredPolymer", "VolumeProfile", "layer_thickness"]

import inspect
from collections import OrderedDict
from time import time

from bumps.parameter import Parameter
import numpy as np
from numpy import exp, hstack, imag, log, ones_like, pi, real, sqrt

try:
    from numpy._core.multiarray import correlate as old_correlate
except ImportError:
    # cruft for numpy < 2
    from numpy.core.multiarray import correlate as old_correlate

from .. import utils
from .layers import Layer

LAMBDA_1 = 1.0 / 6.0  # always assume cubic lattice (1/6) for now
LAMBDA_0 = 1.0 - 2.0 * LAMBDA_1
# Use reverse order for LAMBDA_ARRAY if it is asymmetric since we are using
# it with correlate().
LAMBDA_ARRAY = np.array([LAMBDA_1, LAMBDA_0, LAMBDA_1])
MINLAT = 25
MINBULK = 5
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

    def __init__(
        self,
        thickness=0,
        interface=0,
        name="brush",
        polymer=None,
        solvent=None,
        base_vf=None,
        base=None,
        length=None,
        power=None,
        sigma=None,
    ):
        prefix = name + " "
        self.thickness = Parameter.default(thickness, name=prefix + "thickness")
        self.interface = Parameter.default(interface, name=prefix + "interface")
        self.base_vf = Parameter.default(base_vf, name=prefix + "base_vf")
        self.base = Parameter.default(base, name=prefix + "base")
        self.length = Parameter.default(length, name=prefix + "length")
        self.power = Parameter.default(power, name=prefix + "power")
        self.sigma = Parameter.default(sigma, name=prefix + "sigma")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name
        # Constraints:
        #   base_vf in [0, 1]
        #   base, length, sigma, thickness, interface > 0
        #   base + length + 3*sigma <= thickness

    def parameters(self):
        return {
            "solvent": self.solvent.parameters(),
            "polymer": self.polymer.parameters(),
            "base_vf": self.base_vf,
            "base": self.base,
            "length": self.length,
            "power": self.power,
            "sigma": self.sigma,
        }

    def profile(self, z):
        base_vf, base, length, power, sigma = [
            p.value for p in (self.base_vf, self.base, self.length, self.power, self.sigma)
        ]
        base_vf /= 100.0  # % to fraction
        L0 = base  # if base < thickness else thickness
        L1 = base + length  # if base+length < thickness else thickness-L0
        if length == 0:
            v = np.ones_like(z)
        else:
            v = 1 - ((z - L0) / (L1 - L0)) ** 2
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
        M = Mr + 1j * Mi
        S = Sr + 1j * Si
        try:
            M, S = M[0], S[0]  # Temporary hack
        except Exception:
            pass

        vf = self.profile(Pz)
        Pw, vf = utils.merge_ends(Pw, vf, tol=1e-3)
        P = M * vf + S * (1 - vf)
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
    def __init__(self, thickness=0, interface=0, name="VolumeProfile", material=None, solvent=None, profile=None, **kw):
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
        vars = inspect.getfullargspec(profile)[0]
        # print("vars", vars)
        if inspect.ismethod(profile):
            vars = vars[1:]  # Chop self
        vars = vars[1:]  # Chop z
        # print(vars)
        unused = [k for k in kw.keys() if k not in vars]
        if len(unused) > 0:
            raise TypeError("Profile got unexpected keyword argument '%s'" % unused[0])
        dups = [k for k in vars if k in ("thickness", "interface", "polymer", "solvent", "profile")]
        if len(dups) > 0:
            raise TypeError("Profile has conflicting argument '%s'" % dups[0])
        for k in vars:
            kw.setdefault(k, 0)
        for k, v in kw.items():
            setattr(self, k, Parameter.default(v, name=k))

        self._parameters = vars

    def parameters(self):
        P = {
            "solvent": self.solvent.parameters(),
            "material": self.material.parameters(),
        }
        for k in self._parameters:
            P[k] = getattr(self, k)
        return P

    def render(self, probe, slabs):
        Mr, Mi = self.material.sld(probe)
        Sr, Si = self.solvent.sld(probe)
        M = Mr + 1j * Mi
        S = Sr + 1j * Si
        # M, S = M[0], S[0]  # Temporary hack
        Pw, Pz = slabs.microslabs(self.thickness.value)
        if len(Pw) == 0:
            return
        kw = dict((k, getattr(self, k).value) for k in self._parameters)
        # print(kw)
        phi = self.profile(Pz, **kw)
        try:
            if phi.shape != Pz.shape:
                raise Exception
        except Exception:
            raise TypeError("profile function '%s' did not return array phi(z)" % self.profile.__name__)
        Pw, phi = utils.merge_ends(Pw, phi, tol=1e-3)
        P = M * phi + S * (1 - phi)
        slabs.extend(rho=[real(P)], irho=[imag(P)], w=Pw)
        # slabs.interface(self.interface.value)


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
    dz = z[1] - z[0]
    if 3 * sigma < dz:
        return P
    w = int(3 * sigma / dz)
    G = exp(-0.5 * (np.arange(-w, w + 1) * (dz / sigma)) ** 2)
    full = np.hstack(([P[0]] * w, P, [P[-1]] * w))
    return np.convolve(full, G / np.sum(G), "valid")


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

    def __init__(self, thickness=0, interface=0, name="Mushroom", polymer=None, solvent=None, sigma=0, vf=0, delta=0):
        self.thickness = Parameter.default(thickness, name="Mushroom thickness")
        self.interface = Parameter.default(interface, name="Mushroom interface")
        self.delta = Parameter.default(delta, name="delta")
        self.vf = Parameter.default(vf, name="vf")
        self.sigma = Parameter.default(sigma, name="sigma")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name

    def parameters(self):
        return {
            "solvent": self.solvent.parameters(),
            "polymer": self.polymer.parameters(),
            "delta": self.delta,
            "vf": self.vf,
            "sigma": self.sigma,
            "thickness": self.thickness,
            "interface": self.interface,
        }

    def profile(self, z):
        delta, sigma, vf, thickness = [p.value for p in (self.delta, self.sigma, self.vf, self.thickness)]

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
        M = Mr + 1j * Mi
        S = Sr + 1j * Si
        try:
            # TODO: Fix this hack
            M, S = M[0], S[0]  # Temporary hack
        except Exception:
            pass

        phi = self.profile(Pz)
        Pw, phi = utils.merge_ends(Pw, phi, tol=1e-3)
        P = M * phi + S * (1 - phi)
        Pr, Pi = np.real(P), np.imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)


def MushroomProfile(z, delta=0.1, vf=1.0, sigma=1.0):
    thickness = layer_thickness(z)
    thresh = 1e-10
    base = 3.0 * sigma  # tail is erf, capture 95% of the mixing
    Rg = (thickness - base) / 4.0  # profile ends by ~4 RG, so we can tether these
    keep = (z - base) >= 0.0
    x = (z[keep] - base) / Rg

    """
    mushroom_profile_math has a divide by zero problem at delta=0.
    Fix it by weighted average of the profile above and below a threshold.
    No visual difference when delta is between +-0.001, and there's no
    floating point error until ~+-1e-14.
    """
    if abs(delta) > thresh:
        mushroom_profile = mushroom_math(x, delta, vf)
    else:  # we should RARELY get here
        scale = (delta + thresh) / 2.0 / thresh
        mushroom_profile = scale * mushroom_math(x, thresh, vf) + (1.0 - scale) * mushroom_math(x, -thresh, vf)

    try:
        # make the base connect with the profile
        zextra = z[np.logical_not(keep)]
        base_profile = ones_like(zextra) * mushroom_profile[0]
    except IndexError:
        base_profile = ones_like(z) * mushroom_profile[0]

    return hstack((base_profile, mushroom_profile))


def mushroom_math(x, delta=0.1, vf=0.1):
    """
    new method, rewrite for numerical stability at high delta
    delta=0 causes divide by zero error!! Compensate elsewhere.
    http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
    """
    from scipy.special import erfc, erfcx

    x_half = x / 2.0
    delta_double = 2.0 * delta
    return (
        (
            erfc(x_half)
            - erfcx(delta_double + x_half) / exp(x_half * x_half)
            - erfc(x)
            + ((0.25 - delta * (x + delta_double)) * erfcx(delta_double + x) + delta / SQRT_PI) * 4.0 / exp(x * x)
        )
        * vf
        / (delta_double * erfcx(delta_double))
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
        *phi_b*
            volume fraction of free chains in solution. useful for associating
            grafted films e.g. PS-COOH in Toluene with an SiO2 surface.
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

        l_\text{lat} &= \frac{a^2 m/l}{p_l} \\
        m_\text{lat} &= \frac{(a m/l)^2}{p_l}

    where $l$ is the real polymer's bond length, $m$ is the real segment mass,
    and $a$ is the ratio between molecular weight and radius of gyration at
    theta conditions. The lattice persistence, $p_l$, is:

    .. math::

        p_l = \frac16 \frac{1+1/Z}{1-1/Z}

    with coordination number $Z = 6$ for a cubic lattice, $p_l = .233$.
    """

    def __init__(
        self,
        thickness=0,
        interface=0,
        name="EndTetheredPolymer",
        polymer=None,
        solvent=None,
        chi=0,
        chi_s=0,
        h_dry=None,
        l_lat=1,
        mn=None,
        m_lat=1,
        pdi=1,
        phi_b=0,
    ):
        if interface != 0:
            raise NotImplementedError("interface not yet supported")
        if polymer is None or solvent is None or h_dry is None or mn is None:
            raise TypeError("Need polymer, solvent and profile")

        self.thickness = Parameter.default(thickness, name="SCF thickness")
        self.interface = Parameter.default(interface, name="SCF interface")
        self.chi = Parameter.default(chi, name="Chi")
        self.chi_s = Parameter.default(chi_s, name="Surface chi")
        self.h_dry = Parameter.default(h_dry, name="Dry thickness")
        self.l_lat = Parameter.default(l_lat, name="Lattice layer length")
        self.mn = Parameter.default(mn, name="Num. avg. MW")
        self.m_lat = Parameter.default(m_lat, name="Lattice segegment mass")
        self.pdi = Parameter.default(pdi, name="Dispersity")
        self.phi_b = Parameter.default(phi_b, name="Free polymer conc.")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name

    def parameters(self):
        return {
            "solvent": self.solvent.parameters(),
            "polymer": self.polymer.parameters(),
            "chi": self.chi,
            "chi_s": self.chi_s,
            "h_dry": self.h_dry,
            "l_lat": self.l_lat,
            "mn": self.mn,
            "m_lat": self.m_lat,
            "pdi": self.pdi,
            "phi_b": self.phi_b,
            "thickness": self.thickness,
            "interface": self.interface,
        }

    def profile(self, z):
        return SCFprofile(
            z,
            chi=self.chi.value,
            chi_s=self.chi_s.value,
            h_dry=self.h_dry.value,
            l_lat=self.l_lat.value,
            mn=self.mn.value,
            m_lat=self.m_lat.value,
            pdi=self.pdi.value,
            phi_b=self.phi_b.value,
        )

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
        M = Mr + 1j * Mi
        S = Sr + 1j * Si
        try:
            M, S = M[0], S[0]  # Temporary hack
        except Exception:
            pass

        phi = self.profile(Pz)
        Pw, phi = utils.merge_ends(Pw, phi, tol=1e-3)
        P = M * phi + S * (1 - phi)
        Pr, Pi = np.real(P), np.imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)


def SCFprofile(z, chi=None, chi_s=None, h_dry=None, l_lat=1, mn=None, m_lat=1, phi_b=0, pdi=1, disp=False):
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
    theta = h_dry / l_lat
    segments = mn / m_lat
    sigma = theta / segments

    # solve the self consistent field equations using the cache
    if disp:
        print("\n=====Begin calculations=====\n")
    phi_lat = SCFcache(chi, chi_s, pdi, sigma, phi_b, segments, disp)
    if disp:
        print("\n============================\n")

    # Chop edge effects out
    for x, layer in enumerate(reversed(phi_lat)):
        if abs(layer - phi_b) < 1e-6:
            break
    phi_lat = phi_lat[: -(x + 1)]

    # re-dimensionalize the solution
    layers = len(phi_lat)
    z_end = l_lat * layers
    z_lat = np.linspace(0.0, z_end, num=layers)
    phi = np.interp(z, z_lat, phi_lat, right=phi_b)

    return phi


_SCFcache_dict = OrderedDict()


def SCFcache(chi, chi_s, pdi, sigma, phi_b, segments, disp=False, cache=_SCFcache_dict):
    """Return a memoized SCF result by walking from a previous solution.

    Using an OrderedDict because I want to prune keys FIFO
    """
    try:
        from scipy.optimize import NoConvergence
    except ImportError:
        # cruft for scipy < 1.14, hard breaking change with no warning
        from scipy.optimize.nonlin import NoConvergence
    # prime the cache with a known easy solutions
    if not cache:
        cache[(0, 0, 0, 0.1, 0.1, 0.1)] = SCFsolve(sigma=0.1, phi_b=0.1, segments=50, disp=disp)
        cache[(0, 0, 0, 0, 0.1, 0.1)] = SCFsolve(sigma=0, phi_b=0.1, segments=50, disp=disp)
        cache[(0, 0, 0, 0.1, 0, 0.1)] = SCFsolve(sigma=0.1, phi_b=0, segments=50, disp=disp)

    if disp:
        starttime = time()

    # Try to keep the parameters between 0 and 1. Factors are arbitrary.
    scaled_parameters = (chi, chi_s * 3, pdi - 1, sigma, phi_b, segments / 500)

    # longshot, but return a cached result if we hit it
    if scaled_parameters in cache:
        if disp:
            print("SCFcache hit at:", scaled_parameters)
        phi = cache[scaled_parameters] = cache.pop(scaled_parameters)
        return phi

    # Find the closest parameters in the cache: O(len(cache))

    # Numpy setup
    cached_parameters = tuple(dict.__iter__(cache))
    cp_array = np.array(cached_parameters)
    p_array = np.array(scaled_parameters)

    # Calculate distances to all cached parameters
    deltas = p_array - cp_array  # Parameter space displacement vectors
    closest_index = np.sum(deltas * deltas, axis=1).argmin()

    # Organize closest point data for later use
    closest_cp = cached_parameters[closest_index]
    closest_cp_array = cp_array[closest_index]
    closest_delta = deltas[closest_index]

    phi = cache[closest_cp] = cache.pop(closest_cp)

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

    step = 1.0  # Fractional distance between cached and requested
    dstep = 1.0  # Step size increment
    flag = True

    while flag:
        # end on 1.0 exactly every time
        if step >= 1.0:
            step = 1.0
            flag = False

        # conditional math because, "why risk floating point error"
        if flag:
            p_tup = tuple(closest_cp_array + step * closest_delta)
        else:
            p_tup = scaled_parameters

        if disp:
            print("Parameter step is", step)
            print("current parameters:", p_tup)

        try:
            phi = SCFsolve(
                p_tup[0], p_tup[1] / 3, p_tup[2] + 1, p_tup[3], p_tup[4], p_tup[5] * 500, disp=disp, phi0=phi
            )
        except (NoConvergence, ValueError) as e:
            if isinstance(e, ValueError):
                if str(e) != "array must not contain infs or NaNs":
                    raise
            if disp:
                print("Step failed")
            flag = True  # Reset this so we don't quit if step=1.0 fails
            dstep *= 0.5
            step -= dstep
            if dstep < 1e-5:
                raise RuntimeError("Cache walk appears to be stuck")
        else:  # Belongs to try, executes if no exception is raised
            cache[p_tup] = phi
            dstep *= 1.05
            step += dstep

    if disp:
        print("SCFcache execution time:", round(time() - starttime, 3), "s")

    # keep the cache from consuming all things
    while len(cache) > 100:
        cache.popitem(last=False)

    return phi


def SCFsolve(chi=0, chi_s=0, pdi=1, sigma=None, phi_b=0, segments=None, disp=False, phi0=None, maxiter=30):
    """Solve SCF equations using an initial guess and lattice parameters

    This function finds a solution for the equations where the lattice size
    is sufficiently large.

    The Newton-Krylov solver really makes this one. With gmres, it was faster
    than the other solvers by quite a lot.
    """

    from scipy.optimize import newton_krylov

    if sigma >= 1:
        raise ValueError("Chains that short cannot be squeezed that high")

    if disp:
        starttime = time()

    p_i = SZdist(pdi, segments)

    if phi0 is None:
        # TODO: Better initial guess for chi>.6
        phi0 = default_guess(segments, sigma)
        if disp:
            print("No guess passed, using default phi0: layers =", len(phi0))
    else:
        phi0 = abs(phi0)
        phi0[phi0 > 0.99999] = 0.99999
        if disp:
            print("Initial guess passed: layers =", len(phi0))

    # resizing loop variables
    jac_solve_method = "gmres"
    lattice_too_small = True

    # We tolerate up to 1 ppm deviation from bulk phi
    # when counting layers_near_phi_b
    tol = 1e-6

    def curried_SCFeqns(phi):
        return SCFeqns(phi, chi, chi_s, sigma, segments, p_i, phi_b)

    while lattice_too_small:
        if disp:
            print("Solving SCF equations")

        try:
            with np.errstate(invalid="ignore"):
                phi = abs(
                    newton_krylov(
                        curried_SCFeqns,
                        phi0,
                        verbose=bool(disp),
                        maxiter=maxiter,
                        method=jac_solve_method,
                    )
                )
        except RuntimeError as e:
            if str(e) == "gmres is not re-entrant":
                # Threads are racing to use gmres. Lose the race and use
                # something slower but thread-safe.
                jac_solve_method = "lgmres"
                continue
            else:
                raise

        if disp:
            print("lattice size:", len(phi))

        phi_deviation = abs(phi - phi_b)
        layers_near_phi_b = phi_deviation < tol
        nbulk = np.sum(layers_near_phi_b)
        lattice_too_small = nbulk < MINBULK

        if lattice_too_small:
            # if there aren't enough layers_near_phi_b, grow the lattice 20%
            newlayers = max(1, round(len(phi0) * 0.2))
            if disp:
                print("Growing undersized lattice by", newlayers)
            if nbulk:
                i = np.diff(layers_near_phi_b).nonzero()[0].max()
            else:
                i = phi_deviation.argmin()
            phi0 = np.insert(phi, i, np.linspace(phi[i - 1], phi[i], num=newlayers))

    if nbulk > 2 * MINBULK:
        chop_end = np.diff(layers_near_phi_b).nonzero()[0].max()
        chop_start = chop_end - MINBULK
        i = np.arange(len(phi))
        phi = phi[(i <= chop_start) | (i > chop_end)]

    if disp:
        print("SCFsolve execution time:", round(time() - starttime, 3), "s")

    return phi


_SZdist_dict = OrderedDict()


def SZdist(pdi, nn, cache=_SZdist_dict):
    """Calculate Shultz-Zimm distribution from PDI and number average DP

    Shultz-Zimm is a "realistic" distribution for linear polymers. Numerical
    problems arise when the distribution gets too uniform, so if we find them,
    default to an exact uniform calculation.
    """
    from scipy.special import gammaln

    args = pdi, nn
    if args in cache:
        cache[args] = cache.pop(args)
        return cache[args]

    uniform = False

    if pdi == 1.0:
        uniform = True
    elif pdi < 1.0:
        raise ValueError("Invalid PDI")
    else:
        x = 1.0 / (pdi - 1.0)
        # Calculate the distribution in chunks so we don't waste CPU time
        chunk = 256
        p_ni_list = []
        pdi_underflow = False

        for i in range(max(1, int((100 * nn) / chunk))):
            ni = np.arange(chunk * i + 1, chunk * (i + 1) + 1, dtype=np.float64)
            r = ni / nn
            xr = x * r

            p_ni = exp(log(x / ni) - gammaln(x + 1) + xr * (log(xr) / r - 1))

            pdi_underflow = (p_ni >= 1.0).any()  # catch "too small PDI"
            if pdi_underflow:
                break  # and break out to uniform calculation

            # Stop calculating when species account for less than 1ppm
            keep = (r < 1.0) | (p_ni >= 1e-6)
            if keep.all():
                p_ni_list.append(p_ni)
            else:
                p_ni_list.append(p_ni[keep])
                break
        else:  # Belongs to the for loop. Executes if no break statement runs.
            raise RuntimeError("SZdist overflow")

    if uniform or pdi_underflow:
        # NOTE: rounding here allows nn to be a double in the rest of the logic
        p_ni = np.zeros(int(round(nn)))
        p_ni[-1] = 1.0
    else:
        p_ni = np.concatenate(p_ni_list)
        p_ni /= p_ni.sum()
    cache[args] = p_ni

    if len(cache) > 9000:
        cache.popitem(last=False)

    return p_ni


def default_guess(segments=100, sigma=0.5, phi_b=0.1, chi=0, chi_s=0):
    """Produce an initial guess for phi via analytical approximants.

    For now, a line using numbers from scaling theory
    """
    ss = sqrt(sigma)
    default_layers = int(round(max(MINLAT, segments * ss)))
    default_phi0 = np.linspace(ss, phi_b, num=default_layers)
    return default_phi0


def SCFeqns(phi_z, chi, chi_s, sigma, n_avg, p_i, phi_b=0):
    """System of SCF equation for terminally attached polymers.

    Formatted for input to a nonlinear minimizer or solver.

    The sign convention here on u is "backwards" and always has been.
    It saves a few sign flips, and looks more like Cosgrove's.
    """

    # let the solver go negative if it wants
    phi_z = abs(phi_z)

    # penalize attempts that overfill the lattice
    toomuch = phi_z > 0.99999
    penalty_flag = toomuch.any()
    if penalty_flag:
        penalty = np.where(toomuch, phi_z - 0.99999, 0)
        phi_z[toomuch] = 0.99999

    # calculate new g_z (Boltzmann weighting factors)
    u_prime = log((1.0 - phi_z) / (1.0 - phi_b))
    u_int = 2 * chi * (old_correlate(phi_z, LAMBDA_ARRAY, 1) - phi_b)
    u_int[0] += chi_s
    u_z = u_prime + u_int
    g_z = exp(u_z)

    # normalize g_z for numerical stability
    u_z_avg = np.mean(u_z)
    g_z_norm = g_z / exp(u_z_avg)

    phi_z_new = calc_phi_z(g_z_norm, n_avg, sigma, phi_b, u_z_avg, p_i)

    eps_z = phi_z - phi_z_new

    if penalty_flag:
        np.copysign(penalty, eps_z, penalty)
        eps_z += penalty

    return eps_z


def calc_phi_z(g_z, n_avg, sigma, phi_b, u_z_avg=0, p_i=None):
    if p_i is None:
        segments = n_avg
        uniform = True
    else:
        segments = p_i.size
        uniform = segments == round(n_avg)

    g_zs = Propagator(g_z, segments)

    # for terminally attached chains
    if sigma:
        g_zs_ta = g_zs.ta()

        if uniform:
            c_i_ta = sigma / np.sum(g_zs_ta[:, -1])
            g_zs_ta_ngts = g_zs.ngts_u(c_i_ta)
        else:
            c_i_ta = sigma * p_i / np.sum(g_zs_ta, axis=0)
            g_zs_ta_ngts = g_zs.ngts(c_i_ta)

        phi_z_ta = compose(g_zs_ta, g_zs_ta_ngts, g_z)
    else:
        phi_z_ta = 0

    # for free chains
    if phi_b:
        g_zs_free = g_zs.free()

        if uniform:
            r_i = segments
            c_free = phi_b / r_i
            normalizer = exp(u_z_avg * r_i)
            c_free = c_free * normalizer
            g_zs_free_ngts = g_zs.ngts_u(c_free)
        else:
            r_i = np.arange(1, segments + 1)
            c_i_free = phi_b * p_i / r_i
            normalizer = exp(u_z_avg * r_i)
            c_i_free = c_i_free * normalizer
            g_zs_free_ngts = g_zs.ngts(c_i_free)

        phi_z_free = compose(g_zs_free, g_zs_free_ngts, g_z)
    else:
        phi_z_free = 0

    return phi_z_ta + phi_z_free


def compose(g_zs, g_zs_ngts, g_z):
    prod = g_zs * np.fliplr(g_zs_ngts)
    prod[np.isnan(prod)] = 0
    return np.sum(prod, axis=1) / g_z


class Propagator(object):
    def __init__(self, g_z, segments):
        self.g_z = g_z
        self.shape = int(g_z.size), int(segments)

    def ta(self):
        # terminally attached beginnings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = 0.0
        g_zs[0, 0] = self.g_z[0]
        _calc_g_zs_uniform(self.g_z, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def free(self):
        # free beginnings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = self.g_z
        _calc_g_zs_uniform(self.g_z, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def ngts_u(self, c):
        # free ends of uniform chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c * self.g_z
        _calc_g_zs_uniform(self.g_z, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def ngts(self, c_i):
        # free ends of disperse chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c_i[-1] * self.g_z
        _calc_g_zs(self.g_z, c_i, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def _new(self):
        return np.empty(self.shape, order="F")


try:
    from numba import njit

    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

# USE_NUMBA = False # Uncomment when doing timing tests

if USE_NUMBA:

    @njit("(f8[:], f8[:, :], f8, f8)", cache=True)
    def _calc_g_zs_uniform(g_z, g_zs, f0, f1):
        points, segments = g_zs.shape
        for r in range(segments - 1):
            g_zs[0, r + 1] = (g_zs[0, r] * f0 + g_zs[1, r] * f1) * g_z[0]
            # g_zs[1:-1, r+1] = np.correlate(g_zs[:, r], [f1, f0, f1], 'valid')*g_z[1:-1]
            for k in range(1, points - 1):
                g_zs[k, r + 1] = (g_zs[k, r] * f0 + (g_zs[k - 1, r] + g_zs[k + 1, r]) * f1) * g_z[k]
            g_zs[-1, r + 1] = (g_zs[-2, r] * f1 + g_zs[-1, r] * f0) * g_z[-1]

    @njit("(f8[:], f8[:], f8[:, :], f8, f8)", cache=True)
    def _calc_g_zs(g_z, c_i, g_zs, f0, f1):
        points, segments = g_zs.shape
        for r in range(segments - 1):
            c_ir = c_i[segments - (r + 1) - 1]
            g_zs[0, r + 1] = (g_zs[0, r] * f0 + g_zs[1, r] * f1 + c_ir) * g_z[0]
            # g_zs[1:-1, r+1] = (np.correlate(g_zs[:, r], fir, 'valid') + c_ir)*g_z[1:-1]
            for k in range(1, points - 1):
                g_zs[k, r + 1] = (g_zs[k, r] * f0 + (g_zs[k - 1, r] + g_zs[k + 1, r]) * f1 + c_ir) * g_z[k]
            g_zs[-1, r + 1] = (g_zs[-2, r] * f1 + g_zs[-1, r] * f0 + c_ir) * g_z[-1]

else:

    def _calc_g_zs(g_z, c_i, g_zs, f0, f1):
        coeff = np.array([f1, f0, f1])
        pg_zs = g_zs[:, 0]
        segment_iterator = enumerate(c_i[::-1])
        next(segment_iterator)
        for r, c in segment_iterator:
            g_zs[:, r] = pg_zs = (old_correlate(pg_zs, coeff, 1) + c) * g_z

    def _calc_g_zs_uniform(g_z, g_zs, f0, f1):
        coeff = np.array([f1, f0, f1])
        segments = g_zs.shape[1]
        pg_zs = g_zs[:, 0]
        for r in range(1, segments):
            g_zs[:, r] = pg_zs = old_correlate(pg_zs, coeff, 1) * g_z
