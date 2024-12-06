r"""
Freeform modeling with Chebyshev polynomials

`Chebyshev polynomials <http://en.wikipedia.org/wiki/Chebyshev_polynomials>`_
$T_k$ form a basis set for functions over $[-1, 1]$.  The truncated
interpolating polynomial $P_n$ is a weighted sum of Chebyshev polynomials
up to degree $n$:

.. math::

    f(x) \approx P_n(x) = \sum_{k=0}^n c_i T_k(x)

The interpolating polynomial exactly matches $f(x)$ at the chebyshev
nodes $z_k$ and is near the optimal polynomial approximation to $f$
of degree $n$ under the maximum norm.  For well behaved functions,
the coefficients $c_k$ decrease rapidly, and furthermore are independent
of the degree $n$ of the polynomial.

:class:`FreeformCheby` models the scattering length density profile
of the material within a layer, and :class:`ChebyVF` models the volume
fraction profile of two materials mixed in the layer.

The models can either be defined directly in terms of the Chebyshev
coefficients $c_k$ with *method* = 'direct', or in terms of control
points $(z_k, f(z_k))$ at the Chebyshev nodes :func:`cheby_points`
with *method* = 'interp'.  Bounds on the parameters are easier to
control using 'interp', but the function may oscillate wildly outside
the bounds.  Bounds on the oscillation are easier to control using
'direct', but the shape of the profile is difficult to control.
"""

# TODO: clipping volume fraction to [0, 1] distorts parameter space
# Option 0: clip to [0, 1]
# - Bayesian analysis: parameter values outside the domain will be equally
#   probable out to infinity
# - Newton methods: the fit space is flat outside the domain, which leads
#   to a degenerate hessian.
# - Direct methods: won't fail, but will be subject to random walk
#   performance outside the domain.
# - trivial to implement!
# Option 1: compress (-inf, 0.001] and [0.999, inf) into (0, 0.001], [0.999, 1)
# - won't address any of the problems of clipping
# Option 2: have chisq return inf for points outside the domain
# - Bayesian analysis: correctly assigns probability zero
# - Newton methods: degenerate Hessian outside domain
# - Direct methods: random walk outside domain
# - easy to implement
# Option 3: clip outside domain but add penalty based on amount of clipping
#   A profile based on clipping may have lower chisq than any profile that
#   can be described by a valid model (e.g., by having a sharper transition
#   than would be allowed by the model), leading to a minimum outside D.
#   Adding a penalty constant outside D would help, but there is no constant
#   that works everywhere.  We could use a constant greater than the worst
#   chisq seen so far in D, which can guarantee an arbitrarily low P(x) and
#   a global minimum within D, but for Newton methods, the boundary may still
#   have spurious local minima and objective value now depends on history.
#   Linear compression of profile to fit within the domain would avoid
#   unreachable profile shapes (this is just a linear transform on chebyshev
#   coefficients), and the addition of the penalty value would reduce
#   parameter correlations that result from having transformed parameters
#   resulting in identical profiles.  Returning T = ||A(x)|| from render,
#   with A being a transform that brings the profile within [0, 1], the
#   objective function can return P'(x) = P(x)/(10*(1+sum(T_i)^4) for all
#   slabs i, or P(x) if no slabs return a penalty value.  So long as T is
#   monotonic with increasing badness, with value of 0 within D, and so long
#   as no values of x outside D can generate models that cannot be
#   expressed for any x within D, then any optimizer should return a valid
#   result at the global minimum.  There may still be local minima outside
#   the boundary, so information that the the value is outside the domain
#   still needs to pass through a local optimizer to the fitting program.
#   This approach could be used to transform a box constrained
#   problem to an unconstrained problem using clipping+penalty on the
#   parameter values and removing the need for constrained Newton optimizers.
# - Bayesian analysis: parameters outside D have incorrect probability, but
#   with a sufficiently large penalty, P(x) ~ 0; if the penalty value is
#   too low, details of the correlations outside D may leak into D.
# - Newton methods: Hessian should point back to domain
# - Direct methods: random walk should be biased toward the domain
# - moderately complicated

import numpy as np
from bumps.cheby import cheby_approx, cheby_coeff, cheby_points, cheby_val  # pylint: disable=unused-import
from bumps.parameter import Parameter as Par
from bumps.parameter import to_dict
from numpy import imag, inf, real

from .. import utils
from .layers import Layer


# TODO: add left_sld, right_sld to all layers so that fresnel works
# TODO: access left_sld, right_sld so freeform doesn't need left, right
# TODO: restructure to use vector parameters
# TODO: allow the number of layers to be adjusted by the fit
class FreeformCheby(Layer):
    """
    A freeform section of the sample modeled with Chebyshev polynomials.

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    polynomial orders.
    """

    def __init__(self, thickness=0, interface=0, rho=(), irho=(), name="Cheby", method="interp"):
        if interface != 0:
            raise NotImplementedError("interface not yet supported")
        self.name = name
        self.method = method
        self.thickness = Par.default(thickness, limits=(0, inf), name=name + " thickness")
        self.rho, self.irho = [
            [Par.default(p, name=name + "[%d] %s" % (i, part), limits=limits) for i, p in enumerate(v)]
            for v, part, limits in zip((rho, irho), ("rho", "irho"), ((-inf, inf), (-inf, inf)))
        ]

    def parameters(self):
        """Return parameters used to define layer"""
        return {"thickness": self.thickness, "rho": self.rho, "irho": self.irho}

    def to_dict(self):
        ret = self.parameters()
        ret.update(
            {
                "type": type(self).__name__,
                "name": self.name,
                "method": self.method,
            }
        )
        return to_dict(ret)

    def render(self, probe, slabs):
        """Render slabs for use with the given probe"""
        thickness = self.thickness.value
        Pw, Pz = slabs.microslabs(thickness)
        t = Pz / thickness
        Prho = _profile([p.value for p in self.rho], t, self.method)
        Pirho = _profile([p.value for p in self.irho], t, self.method)
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)


class ChebyVF(Layer):
    r"""
    Material in a solvent

    :Parameters:

        *thickness* : float | Angstrom
            the thickness of the solvent layer
        *interface* : float | Angstrom
            the rms roughness of the solvent surface
        *material* : Material
            the material of interest
        *solvent* : Material
            the solvent or vacuum
        *vf* : [float]
            the control points for volume fraction
        *method* = 'interp' : string | 'direct' or 'interp'
            freeform profile method

    *method* is 'direct' if the *vf* values refer to chebyshev
    polynomial coefficients or 'interp' if *vf* values refer to
    control points located at $z_k$.

    The control point $k$ is located at $z_k \in [0, L]$ for layer
    thickness $L$, as returned by :func:`cheby_points` called with
    n=len(*vf*) and range=\ $[0, L]$.

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi", density=0.965).

    These parameters combine in the following profile formula::

           sld(z) = material.sld * profile(z) + solvent.sld * (1 - profile(z))
    """

    def __init__(self, thickness=0, interface=0, material=None, solvent=None, vf=None, name="ChebyVF", method="interp"):
        if interface != 0:
            raise NotImplementedError("interface not yet supported")
        self.name = name
        self.thickness = Par.default(thickness, name="solvent thickness")
        self.interface = Par.default(interface, name="solvent interface")
        self.solvent = solvent
        self.material = material
        self.vf = [Par.default(p, name="vf[%d]" % i) for i, p in enumerate(vf)]
        self.method = method
        # Constraints:
        #   base_vf in [0, 1]
        #   base, length, sigma, thickness, interface>0
        #   base+length+3*sigma <= thickness

    def parameters(self):
        return {
            "solvent": self.solvent.parameters(),
            "material": self.material.parameters(),
            "vf": self.vf,
        }

    def to_dict(self):
        return to_dict(
            {
                "type": type(self).__name__,
                "name": self.name,
                "method": self.method,
                "thickness": self.thickness,
                "vf": self.vf,
                "material": self.material,
                "solvent": self.solvent,
            }
        )

    def render(self, probe, slabs):
        Mr, Mi = self.material.sld(probe)
        Sr, Si = self.solvent.sld(probe)
        M = Mr + 1j * Mi
        S = Sr + 1j * Si
        try:
            M, S = M[0], S[0]  # Temporary hack
        except Exception:
            pass

        thickness = self.thickness.value
        Pw, Pz = slabs.microslabs(thickness)
        t = Pz / thickness
        vf = _profile([p.value for p in self.vf], t, self.method)
        vf = np.clip(vf, 0, 1)
        Pw, vf = utils.merge_ends(Pw, vf, tol=1e-3)
        P = M * vf + S * (1 - vf)
        Pr, Pi = real(P), imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)


def _profile(c, t, method):
    r"""
    Evaluate the chebyshev approximation c at points x.

    If method is 'direct' then $c_i$ are the coefficients for the chebyshev
    polynomials $T_i$ yielding $P = \sum_i{c_i T_i(x)}$.

    If method is 'interp' then $c_i$ are the values of the interpolated
    function $f$ evaluated at the chebyshev points returned by
    :func:`cheby_points`.
    """
    if method == "interp":
        c = cheby_coeff(c)
    return cheby_val(c, t)
