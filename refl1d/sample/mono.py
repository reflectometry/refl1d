"""
Monotonic spline modeling for free interfaces
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
from bumps.mono import count_inflections, monospline
from bumps.parameter import Constant, to_dict
from bumps.parameter import Function as ParFunction
from bumps.parameter import Parameter as Par
from numpy import asarray, clip, cumsum, diff, hstack, inf, sort

from .. import utils
from .layers import Layer
from .magnetism import BaseMagnetism
from .material import Scatterer


# TODO: add left_sld, right_sld to all layers so that fresnel works
# TODO: access left_sld, right_sld so freeform doesn't need left, right
# TODO: restructure to use vector parameters
# TODO: allow the number of layers to be adjusted by the fit
@dataclass(init=False)
class FreeLayer(Layer):
    """
    A freeform section of the sample modeled with splines.

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    number of control points. The control points can be equally spaced
    in the layers unless rhoz or irhoz are specified. If the z values
    are given, they must be in the range [0, 1].  One control point is
    anchored at either end, so there are two fewer z values than controls
    if z values are given.

    Layers have a slope of zero at the ends, so the automatically blend
    with slabs.
    """

    name: str
    below: Scatterer
    above: Scatterer
    thickness: Par
    z: List[Par]
    rho: List[Par]
    irho: List[Par]
    magnetism: Optional[BaseMagnetism] = None

    def __init__(self, below=None, above=None, thickness=0, z=(), rho=(), irho=(), name="Freeform", magnetism=None):
        self.name = name
        self.below, self.above = below, above
        self.thickness = Par.default(thickness, name=name + " thickness", limits=(0, inf))
        self.interface = Constant(0, name=name + " interface")

        def parvec(vector, name, limits):
            return [Par.default(p, name=name + "[%d]" % i, limits=limits) for i, p in enumerate(vector)]

        self.rho, self.irho, self.z = [
            parvec(v, name + " " + part, limits)
            for v, part, limits in zip((rho, irho, z), ("rho", "irho", "z"), ((-inf, inf), (0, inf), (0, 1)))
        ]
        if len(self.z) != len(self.rho):
            raise ValueError("must have one z for each rho value")
        if len(self.irho) > 0 and len(self.z) != len(self.irho):
            raise ValueError("must have one z for each irho value")

        self.magnetism = magnetism

    def parameters(self):
        return {
            "thickness": self.thickness,
            "rho": self.rho,
            "irho": self.irho,
            "z": self.z,
            "below": self.below.parameters(),
            "above": self.above.parameters(),
            "magnetism": self.magnetism.parameters() if self.magnetism is not None else None,
        }

    def to_dict(self):
        return to_dict(
            {
                "type": type(self).__name__,
                "name": self.name,
                "parameters": self.parameters(),
            }
        )

    def profile(self, Pz, below, above):
        thickness = self.thickness.value
        rbelow, ibelow = below
        rabove, iabove = above
        z = sort([0] + [p.value for p in self.z] + [1]) * thickness

        rho = hstack((rbelow, [p.value for p in self.rho], rabove))
        Prho = monospline(z, rho, Pz)

        if len(self.irho) > 0:
            irho = hstack((ibelow, [p.value for p in self.irho], iabove))
            Pirho = monospline(z, irho, Pz)
        else:
            Pirho = 0 * Prho
        return Prho, Pirho

    def penalty(self):
        dz = diff([p.value for p in self.z])
        return np.sum(dz[dz < 0] ** 2)

    def render(self, probe, slabs):
        below = self.below.sld(probe)
        above = self.above.sld(probe)
        Pw, Pz = slabs.microslabs(self.thickness.value)
        Prho, Pirho = self.profile(Pz, below, above)
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)


def inflections(dx, dy):
    x = hstack((0, cumsum(dx)))
    y = hstack((0, cumsum(dy)))
    return count_inflections(x, y)


@dataclass(init=False)
class FreeInterface(Layer):
    """
    A freeform section of the sample modeled with monotonic splines.

    Layers have a slope of zero at the ends, so the automatically blend
    with slabs.
    """

    name: str
    below: Scatterer
    above: Scatterer
    thickness: Par
    interface: Par
    dz: List[Par]
    dp: List[Par]
    magnetism: Optional[BaseMagnetism] = None
    # inflections: List[Any]

    def __init__(
        self, thickness=0, interface=0, below=None, above=None, dz=None, dp=None, name="Interface", magnetism=None
    ):
        self.name = name
        self.below, self.above = below, above
        self.thickness = Par.default(thickness, limits=(0, inf), name=name + " thickness")
        self.interface = Constant(0, name=name + " interface")

        # Choose reasonable defaults if not given
        if dp is None and dz is None:
            dp = [1] * 5
        if dp is None:
            dp = [1] * len(dz)
        if dz is None:
            dz = [1] * len(dp)
        if len(dz) != len(dp):
            raise ValueError("Need one dz for every dp")

        self.dz = [Par.default(p, name=name + " dz[%d]" % i, limits=(0, inf)) for i, p in enumerate(dz)]
        self.dp = [Par.default(p, name=name + " dp[%d]" % i, limits=(0, inf)) for i, p in enumerate(dp)]
        self.inflections = Par(name=name + " inflections")
        self.inflections.equals(ParFunction(inflections, dx=self.dz, dy=self.dp))
        self.magnetism = magnetism

    def parameters(self):
        return {
            "thickness": self.thickness,
            "interface": self.interface,
            "dz": self.dz,
            "dp": self.dp,
            "below": self.below.parameters(),
            "above": self.above.parameters(),
            "inflections": self.inflections,
            "magnetism": self.magnetism.parameters() if self.magnetism is not None else None,
        }

    def to_dict(self):
        pars = self.parameters()
        pars.pop("inflections")  # derived parameter not needed for save
        return to_dict(
            {
                "type": type(self).__name__,
                "name": self.name,
                "parameters": pars,
            }
        )

    def profile(self, Pz):
        thickness = self.thickness.value
        z, p = [hstack((0, cumsum(asarray([v.value for v in vector], "d")))) for vector in (self.dz, self.dp)]
        if p[-1] == 0:
            p[-1] = 1
        p *= 1 / p[-1]
        # AJC included condition as if z[-1] == 0 then z *= thickness/z[-1] == [nan]*len(z)
        # This then ends with bumps.mono.Monospline adding an extra element as a result of
        # line 42 in bumps.mono.Monospline
        if z[-1] == 0:
            z[-1] = 1
        z *= thickness / z[-1]
        profile = clip(monospline(z, p, Pz), 0, 1)
        return profile

    def render(self, probe, slabs):
        thickness = self.thickness.value

        # TODO: why is provided if it is ignored?
        # interface ignored for FreeInterface
        # interface = self.interface.value

        below_rho, below_irho = self.below.sld(probe)
        above_rho, above_irho = self.above.sld(probe)
        # Pz is the center, Pw is the width
        Pw, Pz = slabs.microslabs(thickness)
        profile = self.profile(Pz)
        Pw, profile = utils.merge_ends(Pw, profile, tol=1e-3)
        Prho = (1 - profile) * below_rho + profile * above_rho
        Pirho = (1 - profile) * below_irho + profile * above_irho
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)


# CRUFT: still working on best rep'n for control point locations
class _FreeInterfaceW(Layer):
    """
    A freeform section of the sample modeled with monotonic splines.

    Layers have a slope of zero at the ends, so the automatically blend
    with slabs.
    """

    def __init__(self, interface=0, below=None, above=None, dz=None, dp=None, name="Interface"):
        self.name = name
        self.below, self.above = below, above
        self.interface = Par.default(interface, limits=(0, inf), name=name + " interface")

        # Choose reasonable defaults if not given
        if dp is None and dz is None:
            dp = [1] * 5
        if dp is None:
            dp = [1] * len(dz)
        if dz is None:
            dz = [10.0 / len(dp)] * len(dp)
        if len(dz) != len(dp):
            raise ValueError("Need one dz for every dp")
        # if len(z) != len(vf)+2:
        #    raise ValueError("Only need vf for interior z, so len(z)=len(vf)+2")
        self.dz = [Par.default(p, name=name + " dz[%d]" % i, limits=(0, inf)) for i, p in enumerate(dz)]
        self.dp = [Par.default(p, name=name + " dp[%d]" % i, limits=(0, inf)) for i, p in enumerate(dp)]

    def _get_thickness(self):
        w = sum(v.value for v in self.dz)
        return Par(w, name=self.name + " thickness")

    def _set_thickness(self, v):
        if v != 0:
            raise ValueError("thickness cannot be set for FreeformInterface")

    thickness = property(_get_thickness, _set_thickness)

    def parameters(self):
        return {
            "interface": self.interface,
            "dz": self.dz,
            "dp": self.dp,
            "below": self.below.parameters(),
            "above": self.above.parameters(),
        }

    def to_dict(self):
        pars = self.parameters()
        pars.pop("inflections")  # derived parameter not needed for save
        return to_dict(
            {
                "type": type(self).__name__,
                "name": self.name,
                "parameters": pars,
            }
        )

    def render(self, probe, slabs):
        interface = self.interface.value
        below_rho, below_irho = self.below.sld(probe)
        above_rho, above_irho = self.above.sld(probe)
        z = hstack((0, cumsum([v.value for v in self.dz])))
        p = hstack((0, cumsum([v.value for v in self.dp])))
        thickness = z[-1]
        if p[-1] == 0:
            p[-1] = 1
        p /= p[-1]
        Pw, Pz = slabs.microslabs(z[-1])
        profile = monospline(z, p, Pz)
        Pw, profile = utils.merge_ends(Pw, profile, tol=1e-3)
        Prho = (1 - profile) * below_rho + profile * above_rho
        Pirho = (1 - profile) * below_irho + profile * above_irho
        slabs.extend(rho=[Prho], irho=[Pirho], w=Pw)
