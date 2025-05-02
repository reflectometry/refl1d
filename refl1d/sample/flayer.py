from dataclasses import dataclass
import inspect
from typing import Dict, Optional, Callable, Union

import numpy as np
from numpy import asarray, broadcast_to, imag, real

from bumps.parameter import Calculation, Constant, Parameter

from refl1d import utils
from .layers import Layer, Stack
from .magnetism import DEFAULT_THETA_M, BaseMagnetism, Magnetism
from .material import SLD


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

    thickness: Parameter
    interface: Constant
    profile: Callable = None
    tol: float = 0
    magnetism: Optional[BaseMagnetism] = None
    name: str = ""

    # Attributes required for serialize/deserialize
    start: SLD = None
    end: SLD = None
    pars: Dict[str, Parameter] = None

    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(
        self,
        thickness=0,
        interface=0,
        profile=None,
        tol=1e-3,
        magnetism=None,
        name=None,
        start=None,
        end=None,
        pars=None,
        **kw,
    ):
        # print(f"building FP with {thickness=} {interface=} {profile=} {name=} {start=} {end=} {pars=} {kw=}")
        if not name:
            name = profile.__name__
        # TODO: maybe we already handle interface with the blend function?
        if float(interface) != 0:
            raise NotImplementedError("interface not yet supported")

        self.name = name
        self.thickness = Parameter.default(thickness, name=name + " thickness")
        self.interface = Constant(float(interface), name=name + " interface")
        self.profile = profile
        self.tol = tol
        self.magnetism = magnetism

        self.start = start if start is not None else SLD(name + " start")
        self.end = end if end is not None else SLD(name + " end")
        self.pars = _parse_parameters(name, profile, kw) if pars is None else pars
        for name, par in self.pars.items():
            setattr(self, name, par)

        # TODO: profile call is duplicated if asking for both rho and irho
        # Fill calculation slots
        self.start.rho.slot = Calculation(
            description="profile rho at z=0",
            function=lambda: real(self._eval([0.0])[0]),
        )
        self.start.irho.slot = Calculation(
            description="profile irho at z=0",
            function=lambda: imag(self._eval([0.0])[0]),
        )
        self.end.rho.slot = Calculation(
            description="profile rho at z=thickness",
            function=lambda: real(self._eval([float(self.thickness)])[0]),
        )
        self.end.irho.slot = Calculation(
            description="profile irho at z=thickness",
            function=lambda: imag(self._eval([float(self.thickness)])[0]),
        )

    def _eval(self, Pz):
        args = {k: float(v) for k, v in self.pars.items()}
        # if self.profile is None:
        #     return np.full_like(Pz, np.nan, dtype=complex)
        return asarray(self.profile(asarray(Pz), **args))

    def parameters(self):
        # TODO: we are not including the calculated parameters
        return {**self.pars, "thickness": self.thickness}

    def render(self, probe, slabs):
        Pw, Pz = slabs.microslabs(self.thickness.value)
        if len(Pw) == 0:
            return
        phi = self._eval(Pz)
        if phi.shape != Pz.shape:
            raise TypeError("profile function '%s' did not return array phi(z)" % self.profile.__name__)
        Pw, phi = utils.merge_ends(Pw, phi, tol=self.tol)
        # P = M*phi + S*(1-phi)
        slabs.extend(rho=[real(phi)], irho=[imag(phi)], w=Pw)
        # TODO: Is the following is sufficient for interfacial roughness?
        # slabs.sigma[0] = float(self.interface)


@dataclass
class FunctionalMagnetism(BaseMagnetism):
    """
    Functional magnetism profile.

    Parameters:

        *profile* the profile function, suitably parameterized

        *tol* is the tolerance for considering values equal

        :class:`refl1d.sample.magnetism.BaseMagnetism` parameters

    The profile function takes a depth vector *z* and returns a magnetism
    vector *rhoM*. For magnetic twist, return a pair of vectors *(rhoM, thetaM)*.
    Constants can be returned for *rhoM* or *thetaM*.  If *thetaM* is not
    provided it defaults to *thetaM=270*.

    See :class:`FunctionalProfile` for a description of the the profile
    function.
    """

    profile: Callable
    tol: float = 1e-3

    # Attributes required for serialize/deserialize
    start: Magnetism = None
    end: Magnetism = None
    thickness: Parameter = None
    pars: Dict[str, Parameter] = None

    BASE_PARS = ("extent", "dead_below", "dead_above", "interface_below", "interface_above")
    magnetic = True

    def __init__(
        self,
        profile=None,
        tol=1e-3,
        name=None,
        start=None,
        end=None,
        thickness=None,
        pars=None,
        **kw,
    ):
        # print(f"building FM with {profile=}, {tol=}, {name=}, {start=}, {end=}, {thickness=}, {pars=}, {kw=}")
        if profile is None:
            raise TypeError("Need profile")
        if not name:
            name = profile.__name__
        # strip magnetism keywords from list of keywords
        magkw = dict((a, kw.pop(a)) for a in set(self.BASE_PARS) & set(kw.keys()))
        BaseMagnetism.__init__(self, name=name, **magkw)
        self.profile = profile
        self.tol = tol

        self.start = start if start is not None else Magnetism(name=name + " start")
        self.end = end if end is not None else Magnetism(name=name + " end")
        self.thickness = Parameter.default(0, name=name + " thickness") if thickness is None else thickness
        self.pars = _parse_parameters(name, profile, kw) if pars is None else pars
        for name, par in self.pars.items():
            setattr(self, name, par)

        # TODO: profile call is duplicated if asking for both rhoM and thetaM
        # Fill calculation slots
        self.start.rhoM.slot = Calculation(
            description="profile magnetic SLD at z=0",
            function=lambda: self._eval([0.0])[0][0],
        )
        self.start.thetaM.slot = Calculation(
            description="profile magnetic angle at z=0",
            function=lambda: self._eval([0.0])[1][0],
        )
        self.end.rhoM.slot = Calculation(
            description="profile magnetic SLD z=thickness",
            function=lambda: self._eval([float(self.thickness)])[0][0],
        )
        self.end.thetaM.slot = Calculation(
            description="profile magnetic angle at z=thickness",
            function=lambda: self._eval([float(self.thickness)])[1][0],
        )
        # print(f"...FM {self.start=}, {self.end=}, {self.end.rhoM=}, {self.thickness=}")

    def set_anchor(self, stack: Stack, index: Union[str, int]):
        """
        Rebuild thickness calculation whenever the sample stack changes. Called from
        :func:`set_magnetism_anchors`.
        """
        stack, start = stack._lookup(index)
        total_thickness = sum(stack[k].thickness for k in range(start, start + self.extent))
        self.thickness.equals(total_thickness - (self.dead_below + self.dead_above))

    def parameters(self):
        # TODO: we are not including the calculated parameters
        return {**BaseMagnetism.parameters(self), **self.pars}

    def _eval(self, Pz):
        args = {k: float(v) for k, v in self.pars.items()}
        Pz = asarray(Pz)
        # if self.profile is None:
        #     return np.full_like(Pz, np.nan), np.full_like(Pz, np.nan)
        P = self.profile(Pz, **args)
        rhoM, thetaM = P if isinstance(P, tuple) else (P, DEFAULT_THETA_M)
        try:
            # rhoM or thetaM may be constant, lists or arrays (but not tuples!)
            rhoM, thetaM = [broadcast_to(v, Pz.shape) for v in (rhoM, thetaM)]
        except ValueError:
            raise TypeError(f"Profile function for '{self.name}' returns incorrect shape")
        return rhoM, thetaM

    def render(self, probe, slabs, thickness, anchor, sigma):
        Pw, Pz = slabs.microslabs(thickness)
        if len(Pw) == 0:
            return
        rhoM, thetaM = self._eval(Pz)

        P = rhoM + thetaM * 0.001j  # combine rhoM/thetaM so they can be merged
        Pw, P = utils.merge_ends(Pw, P, tol=self.tol)
        rhoM, thetaM = P.real, P.imag * 1000  # split out rhoM,thetaM again
        slabs.add_magnetism(anchor=anchor, w=Pw, rhoM=rhoM, thetaM=thetaM, sigma=sigma)

    def __repr__(self):
        return "FunctionalMagnetism(%s)" % self.name


def _parse_parameters(name, profile, kw):
    # Query profile function for the list of arguments
    vars = inspect.getfullargspec(profile)[0]
    # print "vars", vars
    if inspect.ismethod(profile):
        vars = vars[1:]  # Chop self
    vars = vars[1:]  # Chop z
    # print vars
    unused = [k for k in kw.keys() if k not in vars]
    if len(unused) > 0:
        raise TypeError("Profile got unexpected keyword argument '%s'" % unused[0])
    for k in vars:
        kw.setdefault(k, 0)
    pars = {}
    for k, v in kw.items():
        try:
            pv = [Parameter.default(vi, name=f"{name} {k}[{i}]") for i, vi in enumerate(v)]
        except TypeError:
            pv = Parameter.default(v, name=f"{name} {k}")
        pars[k] = pv
    return pars


# TODO: Magnetism can't access the nuclear stack.
def set_magnetism_anchors(stack):
    """
    Find all magnetism objects in the stack and set their anchors.

    This sets the total thickness, which is required to expected magnetism at the end
    of the profile. This function needs to be called for samples that have FunctionalMagnetism
    layers after the sample is built, and whenever the sample structure is changed.
    """
    for k, layer in enumerate(stack):
        if isinstance(layer.magnetism, FunctionalMagnetism):
            layer.magnetism.set_anchor(stack, k)
