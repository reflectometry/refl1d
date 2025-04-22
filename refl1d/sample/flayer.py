import inspect

from bumps.parameter import Calculation, Parameter, to_dict
from numpy import asarray, broadcast_to, imag, real

from refl1d import utils
from .layers import Layer
from .magnetism import DEFAULT_THETA_M, BaseMagnetism, Magnetism
from .material import SLD


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

    RESERVED = ("thickness", "interface", "profile", "tol", "magnetism", "name")

    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(self, thickness=0, interface=0, profile=None, tol=1e-3, magnetism=None, name=None, **kw):
        if not name:
            name = profile.__name__
        if interface != 0:
            raise NotImplementedError("interface not yet supported")
        if profile is None:
            raise TypeError("Need profile")
        self.name = name
        self.thickness = Parameter.default(thickness, name=name + " thickness")
        self.interface = Parameter.default(interface, name=name + " interface")
        self.profile = profile
        self.tol = tol
        self.magnetism = magnetism

        # TODO: maybe make these lazy (and for magnetism below as well)
        rho_start = _LayerLimit(self, isend=False, isrho=True)
        irho_start = _LayerLimit(self, isend=False, isrho=False)
        rho_end = _LayerLimit(self, isend=True, isrho=True)
        irho_end = _LayerLimit(self, isend=True, isrho=False)
        self.start = SLD(name + " start", rho=rho_start, irho=irho_start)
        self.end = SLD(name + " end", rho=rho_end, irho=irho_end)

        _set_parameters(self, name, profile, kw, self.RESERVED)

    def parameters(self):
        P = _get_parameters(self)
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
        Pw, phi = utils.merge_ends(Pw, phi, tol=self.tol)
        # P = M*phi + S*(1-phi)
        slabs.extend(rho=[real(phi)], irho=[imag(phi)], w=Pw)
        # slabs.interface(self.interface.value)


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

    RESERVED = ("profile", "tol", "name", "extent", "dead_below", "dead_above", "interface_below", "interface_above")
    magnetic = True

    def __init__(self, profile=None, tol=1e-3, name=None, **kw):
        if not name:
            name = profile.__name__
        if profile is None:
            raise TypeError("Need profile")
        # strip magnetism keywords from list of keywords
        magkw = dict((a, kw.pop(a)) for a in set(self.RESERVED) & set(kw.keys()))
        BaseMagnetism.__init__(self, name=name, **magkw)
        self.profile = profile
        self.tol = tol

        _set_parameters(self, name, profile, kw, self.RESERVED)
        rhoM_start = _MagnetismLimit(self, isend=False, isrhoM=True)
        rhoM_end = _MagnetismLimit(self, isend=True, isrhoM=True)
        thetaM_start = _MagnetismLimit(self, isend=False, isrhoM=False)
        thetaM_end = _MagnetismLimit(self, isend=True, isrhoM=False)
        self.start = Magnetism(rhoM=rhoM_start, thetaM=thetaM_start)
        self.end = Magnetism(rhoM=rhoM_end, thetaM=thetaM_end)
        self.anchor = None

    def set_anchor(self, stack, index):
        self.anchor = (stack, index)

    # TODO: is there a sane way of computing magnetism thickness in advance?
    def _calc_thickness(self):
        if self.anchor is None:
            raise ValueError("Need layer.magnetism.set_anchor(stack, layer) to compute magnetic thickness.")
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
        Pw, P = utils.merge_ends(Pw, P, tol=self.tol)
        rhoM, thetaM = P.real, P.imag * 1000  # split out rhoM,thetaM again
        slabs.add_magnetism(anchor=anchor, w=Pw, rhoM=rhoM, thetaM=thetaM, sigma=sigma)

    def __repr__(self):
        return "FunctionalMagnetism(%s)" % self.name


def _set_parameters(self, name, profile, kw, reserved):
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
        setattr(self, k, pv)
    self._parameters = vars


def _get_parameters(self):
    return {k: getattr(self, k) for k in self._parameters}


def _get_values(self):
    vals = {}
    for k in self._parameters:
        v = getattr(self, k)
        if isinstance(v, list):
            vals[k] = asarray([vk.value for vk in v], "d")
        else:
            vals[k] = v.value
    return vals


class _LayerLimit(Calculation):
    def __init__(self, flayer, isend=True, isrho=True):
        self.description = f"{'rho' if isrho else 'irho'} Layer Limit, isend={isend}"
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
    def __init__(self, flayer, isend=True, isrhoM=True):
        self.description = f"{'rhoM' if isrhoM else 'thetaM'} Magnetism Limit, isend={isend}"
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
