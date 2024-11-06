from dataclasses import dataclass
import inspect
import numbers
from typing import Dict, Literal, Optional, Tuple
import inspect

from numpy import real, imag, asarray, broadcast_to

from bumps.parameter import Parameter, to_dict, Calculation, Constant, Expression
from bumps.util import schema_config
from refl1d.material import SLD
from refl1d.model import Layer, Stack
from refl1d.magnetism import BaseMagnetism, Magnetism, DEFAULT_THETA_M
from refl1d import util


@dataclass
class FunctionalProfile(Layer):
    """
    Generic profile function

    Parameters:

        *thickness* the thickness of the layer

        *interface* the roughness of the surface [not implemented]

        *profile* the profile function, suitably parameterized

        *tol* is the tolerance for considering values equal

        *magnetism* magnetic profile associated with the layer

        *name* is the layer name

    The profile function takes a depth vector *z* returns a density vector
    *rho*. For absorbing profiles, return complex vector *rho + irho*1j*.

    Fitting parameters are the available named arguments to the function.
    The first argument is a depth vector, which is the array of depths at
    which the profile is to be evaluated.  It is guaranteed to be increasing,
    with step size 2*z[0].

    Initial values for the function parameters can be given using name=value.
    These values can be scalars or fitting parameters.  The function will
    be called with the current parameter values as arguments.  The layer
    thickness can be computed as :func:`layer_thickness`.

    There is no mechanism for querying the larger profile to determine the
    value of the *rho* at the layer boundaries.  If needed, this information
    will have to be communicated through shared parameters.  For example::

        L1 = SLD('L1', rho=2.07)
        L3 = SLD('L3', rho=4)
        def linear(z, rhoL, rhoR):
            rho = z * (rhoR-rhoL)/(z[-1]-z[0]) + rhoL
            return rho
        profile = FunctionalProfile(100, 0, profile=linear,
                                    rhoL=L1.rho, rhoR=L3.rho)
        sample = L1 | profile | L3
    """

    name: str
    thickness: Parameter
    interface: Constant
    profile_source: str = None
    profile_params: Dict[str, Parameter] = None
    tol: float = 0
    magnetism: Optional[BaseMagnetism]
    rho_start: Parameter = None
    rho_end: Parameter = None
    irho_start: Parameter = None
    irho_end: Parameter = None

    RESERVED = ("thickness", "interface", "profile", "tol", "magnetism", "name")

    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(
        self,
        thickness=0,
        interface=0,
        profile=None,
        profile_source=None,
        tol=1e-3,
        magnetism=None,
        name=None,
        profile_params=None,
        rho_start=None,
        rho_end=None,
        irho_start=None,
        irho_end=None,
        **kw,
    ):
        if not name:
            name = profile.__name__
        if float(interface) != 0:
            raise NotImplementedError("interface not yet supported")

        if profile_source is not None:
            self.profile_source = profile_source
            import numpy as np
            import scipy

            ctx = {"np": np, "scipy": scipy}
            output = {}
            exec(profile_source, ctx, output)
            # assume that the function definition is the only thing in the source
            profile_name = list(output.keys())[0]
            self.profile = output[profile_name]
        elif profile is not None:
            self.profile = profile
            self.profile_source = inspect.getsource(profile)
            profile_name = profile.__name__
        else:
            raise TypeError("Need profile or profile_source")
        if not name:
            name = profile_name

        self.name = name
        self.thickness = Parameter.default(thickness, name=name + " thickness")
        self.interface = Constant(interface, name=name + " interface")
        self.rho_start = Parameter.default(rho_start, name=name + " rho_start")
        self.rho_end = Parameter.default(rho_end, name=name + " rho_end")
        self.irho_start = Parameter.default(irho_start, name=name + " irho_start")
        self.irho_end = Parameter.default(irho_end, name=name + " irho_end")
        self.tol = tol
        self.magnetism = magnetism

        if profile_params is not None:
            self.profile_params = profile_params
        else:
            _set_parameters(self, name, self.profile, kw, self.RESERVED)

        # TODO: maybe make these lazy (and for magnetism below as well)
        self._set_ends()

    def _set_ends(self):
        rho_start = Calculation(self.name + " rho_start")
        rho_start.set_function(lambda: real(self.profile(asarray([0.0]), **_get_values(self))[0]))
        irho_start = Calculation(self.name + " irho_start")
        irho_start.set_function(lambda: imag(self.profile(asarray([0.0]), **_get_values(self))[0]))
        rho_end = Calculation(self.name + " rho_end")
        rho_end.set_function(lambda: real(self.profile(asarray([self.thickness.value]), **_get_values(self))[0]))
        irho_end = Calculation(self.name + " irho_end")
        irho_end.set_function(lambda: imag(self.profile(asarray([self.thickness.value]), **_get_values(self))[0]))
        self.rho_start.slot = rho_start
        self.irho_start.slot = irho_start
        self.rho_end.slot = rho_end
        self.irho_end.slot = irho_end

    @property
    def start(self):
        return SLD(self.name + " start", rho=self.rho_start, irho=self.irho_start)

    @property
    def end(self):
        return SLD(self.name + " end", rho=self.rho_end, irho=self.irho_end)

    def parameters(self):
        P = {}
        P.update(self.profile_params)
        P["thickness"] = self.thickness
        # P['interface'] = self.interface
        return P

    def to_dict(self):
        return to_dict(
            {
                "type": type(self).__name__,
                "name": self.name,
                "thickness": self.thickness,
                "interface": self.interface,
                "profile": self.profile,
                "parameters": _get_parameters(self),
                "tol": self.tol,
                "magnetism": self.magnetism,
            }
        )

    def render(self, probe, slabs):
        Pw, Pz = slabs.microslabs(self.thickness.value)
        if len(Pw) == 0:
            return
        # print kw
        # TODO: always return rho, irho from profile function
        # return value may be a constant for rho or irho
        phi = asarray(self.profile(Pz, **_get_values(self)))
        if phi.shape != Pz.shape:
            raise TypeError("profile function '%s' did not return array phi(z)" % self.profile.__name__)
        Pw, phi = util.merge_ends(Pw, phi, tol=self.tol)
        # P = M*phi + S*(1-phi)
        slabs.extend(rho=[real(phi)], irho=[imag(phi)], w=Pw)
        # slabs.interface(self.interface.value)


@dataclass
class FunctionalMagnetism(BaseMagnetism):
    """
    Functional magnetism profile.

    Parameters:

        *profile* the profile function, suitably parameterized

        *tol* is the tolerance for considering values equal

        :class:`refl1d.magnetism.BaseMagnetism` parameters

    The profile function takes a depth vector *z* and returns a magnetism
    vector *rhoM*. For magnetic twist, return a pair of vectors *(rhoM, thetaM)*.
    Constants can be returned for *rhoM* or *thetaM*.  If *thetaM* is not
    provided it defaults to *thetaM=270*.

    See :class:`FunctionalProfile` for a description of the the profile
    function.
    """

    name: str
    profile_source: str = None
    tol: float = 1e-3
    rhoM_start: Parameter = None  # Calculation objects...
    thetaM_start: Parameter = None
    rhoM_end: Parameter = None
    thetaM_end: Parameter = None
    thickness: Parameter = None
    profile_params: Dict[str, Parameter] = None

    RESERVED = ("profile", "tol", "name", "extent", "dead_below", "dead_above", "interface_below", "interface_above")
    magnetic = True

    def __init__(
        self,
        profile_source=None,
        profile=None,
        tol=1e-3,
        name=None,
        thickness=None,
        profile_params=None,
        rhoM_start=None,
        thetaM_start=None,
        rhoM_end=None,
        thetaM_end=None,
        **kw,
    ):
        if profile_source is not None:
            self.profile_source = profile_source
            import numpy as np
            import scipy

            ctx = {"np": np, "scipy": scipy}
            output = {}
            exec(profile_source, ctx, output)
            # assume that the function definition is the only thing in the source
            profile_name = list(output.keys())[0]
            self.profile = output[profile_name]
        elif profile is not None:
            self.profile = profile
            self.profile_source = inspect.getsource(profile)
            profile_name = profile.__name__
        else:
            raise TypeError("Need profile or profile_source")
        if not name:
            name = profile_name

        self.name = name
        self.tol = tol
        # strip magnetism keywords from list of keywords
        magkw = dict((a, kw.pop(a)) for a in set(self.RESERVED) & set(kw.keys()))
        BaseMagnetism.__init__(self, name=name, **magkw)
        self.thickness = Parameter.default(thickness, name=name + " thickness")
        self.rhoM_start = Parameter.default(rhoM_start, name=name + " rhoM_start")
        self.thetaM_start = Parameter.default(thetaM_start, name=name + " thetaM_start")
        self.rhoM_end = Parameter.default(rhoM_end, name=name + " rhoM_end")
        self.thetaM_end = Parameter.default(thetaM_end, name=name + " thetaM_end")

        if profile_params is not None:
            self.profile_params = profile_params
        else:
            _set_parameters(self, name, self.profile, kw, self.RESERVED)
        self._set_ends()

    def set_anchor(self, stack, index):
        stack, start = stack._lookup(index)
        thickness_params = []
        for k in range(start, start + self.extent):
            thickness_params.append(stack[k].thickness)
        total_expression = sum(thickness_params) - self.dead_below - self.dead_above
        self.thickness = total_expression

    def _set_ends(self):
        rhoM_start = Calculation(self.name + " rhoM_start")

        def rhoM_start_func():
            P = self.profile(asarray([0.0]), **_get_values(self))
            return P[0][0] if isinstance(P, tuple) else P[0]

        rhoM_start.set_function(rhoM_start_func)
        thetaM_start = Calculation(self.name + " thetaM_start")

        def thetaM_start_func():
            P = self.profile(asarray([0.0]), **_get_values(self))
            return P[1][0] if isinstance(P, tuple) else DEFAULT_THETA_M

        thetaM_start.set_function(thetaM_start_func)
        rhoM_end = Calculation(self.name + " rhoM_end")

        def rhoM_end_func():
            P = self.profile(asarray([self.thickness.value]), **_get_values(self))
            return P[0][0] if isinstance(P, tuple) else P[0]

        rhoM_end.set_function(rhoM_end_func)
        thetaM_end = Calculation(self.name + " thetaM_end")

        def thetaM_end_func():
            P = self.profile(asarray([self.thickness.value]), **_get_values(self))
            return P[1][0] if isinstance(P, tuple) else DEFAULT_THETA_M

        thetaM_end.set_function(thetaM_end_func)
        self.rhoM_start.slot = rhoM_start
        self.thetaM_start.slot = thetaM_start
        self.rhoM_end.slot = rhoM_end
        self.thetaM_end.slot = thetaM_end

    @property
    def start(self):
        return Magnetism(rhoM=self.rhoM_start, thetaM=self.thetaM_start)

    @property
    def end(self):
        return Magnetism(rhoM=self.rhoM_end, thetaM=self.thetaM_end)

    # TODO: is there a sane way of computing magnetism thickness in advance?
    def _calc_thickness(self):
        if self.anchor is None:
            raise ValueError("Need layer.magnetism.set_anchor(stack, layer) to compute" " magnetic thickness.")
        stack, index = self.anchor
        stack, start = stack._lookup(index)
        total = 0
        for k in range(start, start + self.extent):
            total += stack[k].thickness.value
        total -= self.dead_below.value + self.dead_above.value
        return total

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(_get_parameters(self))
        return parameters

    def to_dict(self):
        ret = BaseMagnetism.to_dict(self)
        ret.update(
            to_dict(
                {
                    "profile": self.profile,
                    "parameters": _get_parameters(self),
                    "tol": self.tol,
                }
            )
        )

    def render(self, probe, slabs, thickness, anchor, sigma):
        Pw, Pz = slabs.microslabs(thickness)
        if len(Pw) == 0:
            return
        P = self.profile(Pz, **_get_values(self))

        rhoM, thetaM = P if isinstance(P, tuple) else (P, DEFAULT_THETA_M)
        try:
            # rhoM or thetaM may be constant, lists or arrays (but not tuples!)
            rhoM, thetaM = [broadcast_to(v, Pz.shape) for v in (rhoM, thetaM)]
        except ValueError:
            raise TypeError("profile function '%s' did not return array rhoM(z)" % self.profile.__name__)
        P = rhoM + thetaM * 0.001j  # combine rhoM/thetaM so they can be merged
        Pw, P = util.merge_ends(Pw, P, tol=self.tol)
        rhoM, thetaM = P.real, P.imag * 1000  # split out rhoM,thetaM again
        slabs.add_magnetism(anchor=anchor, w=Pw, rhoM=rhoM, thetaM=thetaM, sigma=sigma)

    def __repr__(self):
        return "FunctionalMagnetism(%s)" % self.name


def _set_parameters(self, name, profile, kw, reserved):
    # Query profile function for the list of arguments
    profile_params = {}
    vars = inspect.getfullargspec(profile)[0]
    # print "vars", vars
    if inspect.ismethod(profile):
        vars = vars[1:]  # Chop self
    vars = vars[1:]  # Chop z
    # print vars
    unused = [k for k in kw.keys() if k not in vars]
    if len(unused) > 0:
        raise TypeError("Profile got unexpected keyword argument '%s'" % unused[0])
    dups = [k for k in vars if k in reserved]
    if len(dups) > 0:
        raise TypeError("Profile has conflicting argument %r" % dups[0])
    for k in vars:
        kw.setdefault(k, 0)
    for k, v in kw.items():
        try:
            pv = [Parameter.default(vi, name=f"{name} {k}[{i}]") for i, vi in enumerate(v)]
        except TypeError:
            pv = Parameter.default(v, name=f"{name} {k}")
        profile_params[k] = pv
    self.profile_params = profile_params


def _get_parameters(self):
    return self.profile_params
    # return {k: getattr(self, k) for k in self.vars}


def _get_values(self):
    vals = {}
    for k, v in self.profile_params.items():
        # v = getattr(self, k)
        if isinstance(v, list):
            vals[k] = asarray([vk.value for vk in v], "d")
        else:
            vals[k] = v.value
    return vals


@dataclass(init=False)
class _LayerLimit(Calculation):
    flayer: FunctionalProfile
    isend: bool
    isrho: bool

    def __init__(self, flayer, isend=True, isrho=True, description=None):
        if description is None:
            description = f"{'rho' if isrho else 'irho'} Layer Limit, isend={isend}"
        self.description = description
        self.flayer = flayer
        self.isend = isend
        self.isrho = isrho
        self.name = str(flayer) + self._tag

    @property
    def _tag(self):
        return (".rho_" if self.isrho else ".irho_") + ("end" if self.isend else "start")

    def parameters(self):
        return []

    def _function(self):
        z = asarray([0.0, self.flayer.thickness.value])
        P = self.flayer.profile(asarray(z), **_get_values(self.flayer))
        index = 1 if self.isend else 0
        return real(P[index]) if self.isrho else imag(P[index])

    def __repr__(self):
        return repr(self.flayer) + self._tag


class _MagnetismLimit(Calculation):
    def __init__(self, flayer, isend=True, isrhoM=True, description=None):
        if description is None:
            description = f"{'rhoM' if isrhoM else 'thetaM'} Magnetism Limit, isend={isend}"
        self.description = description
        self.flayer = flayer
        self.isend = isend
        self.isrhoM = isrhoM
        self.name = str(flayer) + self._tag

    @property
    def _tag(self):
        return (".rhoM_" if self.isrhoM else ".thetaM_") + ("end" if self.isend else "start")

    def parameters(self):
        return []

    def _function(self):
        zmax = self.flayer._calc_thickness()
        z = asarray([0.0, zmax])
        P = self.flayer.profile(z, **_get_values(self.flayer))
        rhoM, thetaM = P if isinstance(P, tuple) else (P, DEFAULT_THETA_M)
        rhoM, thetaM = [broadcast_to(v, z.shape) for v in (rhoM, thetaM)]
        index = -1 if self.isend else 0
        return rhoM[index] if self.isrhoM else thetaM[index]

    def __repr__(self):
        return repr(self.flayer) + self._tag
