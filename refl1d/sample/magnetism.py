# Author: Paul Kienzle
r"""
Magnetic modeling for 1-D reflectometry.

Magnetic properties are tied to the structural description of the
but only loosely.

There may be dead regions near the interfaces of magnetic materials.

Magnetic behaviour may be varying in complex ways within and
across structural boundaries.  For example, the ma
Indeed, the pattern may continue
across spacer layers, going to zero in the magnetically dead
region and returning to its long range variation on entry to
the next magnetic layer.  Magnetic multilayers may exhibit complex
magnetism throughout the repeated section while the structural
components are fixed.

The scattering behaviour is dependent upon net field strength relative to
polarization direction.   This arises from three underlying quantities:
the strength of the individual dipole moments in the layer, the degree
of alignment of these moments, and the net direction of the alignment.
The strength of the dipole moment depends on the details of the electronic
structure, so unlike the nuclear scattering potential, it cannot be readily
determined from material composition.  Similarly, net magnetization
depends on the details of the magnetic domains within the material, and
cannot readily be determined from first principles.  The interaction
potential of the net magnetic moment depends on the alignment of the field
with respect to the beam, with a net scattering length density of
:math:`\rho_M \cos(\theta_M)`.  Clearly the scattering measurement will
not be able to distinguish between a reduced net magnetic strength
:math:`\rho_M` and a change in orientation :math:`\theta_M` for an
individual measurement, as should be apparent from the correlated
uncertainty plot produced when both parameters are fit.

Magnetism support is split into two parts: describing the layers
and anchoring them to the structure.
"""

from dataclasses import dataclass
from typing import Union, Dict, Literal, List

from bumps.parameter import Parameter, flatten, to_dict
from bumps.mono import monospline
import numpy as np

from refl1d.sample.reflectivity import BASE_GUIDE_ANGLE as DEFAULT_THETA_M


@dataclass
class BaseMagnetism:
    """
    Magnetic properties of the layer.

    Magnetism is attached to set of nuclear layers by setting the
    *magnetism* property of the first layer to the rendered for the
    magnetic profile, and setting *extent* to the number of nuclear
    layers attached to the magnetism object.

    *dead_below* and *dead_above* are dead regions within the magnetic
    extent, which allow you to shift the magnetic interfaces relative
    to the nuclear interfaces.

    *interface_below* and *interface_above* are the interface widths
    for the magnetic layer, which default to the interface widths for
    the corresponding nuclear layers if no interfaces are specified.
    For consecutive layers, only *interface_above* is used; any value
    for *interface_below* is ignored.
    """

    name: str
    extent: float
    dead_below: Union[Parameter, Literal[None]]
    dead_above: Union[Parameter, Literal[None]]
    interface_below: Union[Parameter, Literal[None]]
    interface_above: Union[Parameter, Literal[None]]

    def __init__(self, extent=1, dead_below=0, dead_above=0, interface_below=None, interface_above=None, name="LAYER"):
        self.dead_below = Parameter.default(dead_below, limits=(0, None), name=name + " deadM below")
        self.dead_above = Parameter.default(dead_above, limits=(0, None), name=name + " deadM above")
        if interface_below is not None:
            interface_below = Parameter.default(interface_below, limits=(0, None), name=name + " interfaceM below")
        if interface_above is not None:
            interface_above = Parameter.default(interface_above, limits=(0, None), name=name + " interfaceM above")
        self.interface_below = interface_below
        self.interface_above = interface_above
        self.name = name
        self.extent = extent

    def parameters(self) -> Dict[str, Union[Parameter, List[Parameter], Literal[None]]]:
        return {
            "dead_below": self.dead_below,
            "dead_above": self.dead_above,
            "interface_below": self.interface_below,
            "interface_above": self.interface_above,
        }

    def to_dict(self):
        return to_dict(
            {
                "type": type(self).__name__,
                "name": self.name,
                "extent": self.extent,
                "dead_below": self.dead_below,
                "dead_above": self.dead_above,
                "interface_below": self.interface_below,
                "interface_above": self.interface_above,
            }
        )

    def set_layer_name(self, name):
        """
        Update the names of the magnetic parameters with the name of the
        layer if it has not already been set.  This is necessary since we don't
        know the layer name until after we have constructed the magnetism object.
        """
        if self.name == "LAYER":
            for p in flatten(self.parameters()):
                p.name = p.name.replace("LAYER", name)
            self.name = name


@dataclass(init=False)
class Magnetism(BaseMagnetism):
    """
    Region of constant magnetism.

    *rhoM* is the magnetic SLD the layer. Default is :code:`rhoM=0`.

    *thetaM* is the magnetic angle for the layer. Default is :code:`thetaM=270`.

    *name* is the base name for the various layer parameters.

    *extent* defines the number of nuclear layers covered by the magnetic layer.

    *dead_above* and *dead_below* define magnetically dead layers at the
    nuclear boundaries.  These can be negative if magnetism extends beyond
    the nuclear boundary.

    *interface_above* and *interface_below* define the magnetic interface
    at the boundaries, if it is different from the nuclear interface.
    """

    rhoM: Parameter
    thetaM: Parameter

    def __init__(self, rhoM: Union[float, Parameter] = 0, thetaM=DEFAULT_THETA_M, name="LAYER", **kw):
        BaseMagnetism.__init__(self, name=name, **kw)
        self.rhoM = Parameter.default(rhoM, name=name + " rhoM")
        self.thetaM = Parameter.default(thetaM, limits=(0, 360), name=name + " thetaM")

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM)
        return parameters

    def to_dict(self):
        result = BaseMagnetism.to_dict(self)
        result["rhoM"] = to_dict(self.rhoM)
        result["thetaM"] = to_dict(self.thetaM)
        return result

    def render(self, probe, slabs, thickness, anchor, sigma):
        slabs.add_magnetism(
            anchor=anchor, w=[thickness], rhoM=[self.rhoM.value], thetaM=[self.thetaM.value], sigma=sigma
        )

    def __str__(self):
        return "magnetism(%g)" % self.rhoM.value

    def __repr__(self):
        return "Magnetism(rhoM=%g, thetaM=%g)" % (self.rhoM.value, self.thetaM.value)


@dataclass(init=False)
class MagnetismStack(BaseMagnetism):
    """
    Magnetic slabs within a magnetic layer.

    *weight* is the relative thickness of each layer relative to the nuclear
    stack to which it is anchored.  Weights are automatically normalized to 1.
    Default is :code:`weight=[1]` equal size layers.

    *rhoM* is the magnetic SLD for each layer. Default is :code:`rhoM=[0]`
    for shared magnetism in all the layers.

    *thetaM* is the magnetic angle for each layer.  Default is
    :code:`thetaM=[270]` for no magnetic twist.

    **Not yet implemented.**
    *interfaceM* is the magnetic interface for all but the last layer.  Default
    is :code:`interfaceM=[0]` for equal width interfaces in all layers.

    *name* is the base name for the various layer parameters.

    *extent* defines the number of nuclear layers covered by the magnetic layer.

    *dead_above* and *dead_below* define magnetically dead layers at the
    nuclear boundaries.  These can be negative if magnetism extends beyond
    the nuclear boundary.

    *interface_above* and *interface_below* define the magnetic interface
    at the boundaries, if it is different from the nuclear interface.
    """

    weight: List[Parameter]
    rhoM: List[Parameter]
    thetaM: List[Parameter]

    def __init__(self, weight=None, rhoM=None, thetaM=None, interfaceM=None, name="LAYER", **kw):
        weight_n = 0 if weight is None else len(weight)
        rhoM_n = 0 if rhoM is None else len(rhoM)
        thetaM_n = 0 if thetaM is None else len(thetaM)
        interfaceM_n = 0 if interfaceM is None else (len(interfaceM) + 1)
        n = max(weight_n, rhoM_n, thetaM_n, interfaceM_n)

        if n == 0:
            raise ValueError("Must specify one of weight, rhoM, thetaM or interfaceM as vector")
        if (
            (weight_n > 1 and weight_n != n)
            or (rhoM_n > 1 and rhoM_n != n)
            or (thetaM_n > 1 and thetaM_n != n)
            or (interfaceM_n > 1 and interfaceM_n != n)
        ):
            raise ValueError("Inconsistent lengths for weight, rhoM, thetaM and interfaceM")

        # TODO: intefaces need to be implemented in profile.add_magnetism
        if interfaceM is not None:
            raise NotImplementedError("Doesn't yet support magnetic interfaces")

        if weight is None:
            weight = [1]
        if rhoM is None:
            rhoM = [0]
        if thetaM is None:
            thetaM = [DEFAULT_THETA_M]
        # if interfaceM is None:
        #    interfaceM = [0]

        BaseMagnetism.__init__(self, name=name, **kw)
        self.weight = [Parameter.default(v, name=name + " weight[%d]" % i) for i, v in enumerate(weight)]
        self.rhoM = [Parameter.default(v, name=name + " rhoM[%d]" % i) for i, v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name + " thetaM[%d]" % i) for i, v in enumerate(thetaM)]
        # self.interfaceM = [Parameter.default(v, name=name+" interfaceM[%d]"%i)
        #                   for i, v in enumerate(interfaceM)]

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(
            rhoM=self.rhoM,
            thetaM=self.thetaM,
            # interfaceM=self.interfaceM,
            weight=self.weight,
        )
        return parameters

    def to_dict(self):
        result = BaseMagnetism.to_dict(self)
        result["weight"] = to_dict(self.weight)
        result["rhoM"] = to_dict(self.rhoM)
        result["thetaM"] = to_dict(self.thetaM)
        # result['interfaceM'] = to_dict(self.interfaceM)
        return result

    def render(self, probe, slabs, thickness, anchor, sigma):
        w = np.array([p.value for p in self.weight])
        w *= thickness / np.sum(w)
        rhoM = [p.value for p in self.rhoM]
        thetaM = [p.value for p in self.thetaM]
        # interfaceM = [p.value for p in self.interfaceM]
        if len(rhoM) == 1:
            rhoM = [rhoM[0]] * len(w)
        if len(thetaM) == 1:
            thetaM = [thetaM[0]] * len(w)
        # if len(interfaceM) == 1:
        #    interfaceM = [interfaceM[0]]*(len(w)-1)

        slabs.add_magnetism(
            anchor=anchor,
            w=w,
            rhoM=rhoM,
            thetaM=thetaM,
            # interfaceM=interfaceM,
            sigma=sigma,
        )

    def __str__(self):
        return "MagnetismStack(%d)" % (len(self.rhoM))

    def __repr__(self):
        return "MagnetismStack"


@dataclass(init=False)
class MagnetismTwist(BaseMagnetism):
    """
    Linear change in magnetism throughout layer.

    *rhoM* contains the *(left, right)* values for the magnetic scattering
    length density.  The number of steps is determined by the model *dz*.

    *thetaM* contains the *(left, right)* values for the magnetic angle.

    *name* is the base name for the various layer parameters.

    *extent* defines the number of nuclear layers covered by the magnetic layer.

    *dead_above* and *dead_below* define magnetically dead layers at the
    nuclear boundaries.  These can be negative if magnetism extends beyond
    the nuclear boundary.

    *interface_above* and *interface_below* define the magnetic interface
    at the boundaries, if it is different from the nuclear interface.
    """

    rhoM: List[Parameter]
    thetaM: List[Parameter]

    magnetic = True

    def __init__(self, rhoM=(0, 0), thetaM=(DEFAULT_THETA_M, DEFAULT_THETA_M), name="LAYER", **kw):
        BaseMagnetism.__init__(self, name=name, **kw)
        self.rhoM = [Parameter.default(v, name=name + " rhoM[%d]" % i) for i, v in enumerate(rhoM)]
        self.thetaM = [Parameter.default(v, name=name + " thetaM[%d]" % i) for i, v in enumerate(thetaM)]

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM)
        return parameters

    def to_dict(self):
        result = BaseMagnetism.to_dict(self)
        result["rhoM"] = to_dict(self.rhoM)
        result["thetaM"] = to_dict(self.thetaM)
        return result

    def render(self, probe, slabs, thickness, anchor, sigma):
        w, z = slabs.microslabs(thickness)
        rhoM = np.linspace(self.rhoM[0].value, self.rhoM[1].value, len(z))
        thetaM = np.linspace(self.thetaM[0].value, self.thetaM[1].value, len(z))
        slabs.add_magnetism(anchor=anchor, w=w, rhoM=rhoM, thetaM=thetaM, sigma=sigma)

    def __str__(self):
        return "twist(%g->%g)" % (self.rhoM[0].value, self.rhoM[1].value)

    def __repr__(self):
        return "MagneticTwist"


@dataclass(init=False)
class FreeMagnetism(BaseMagnetism):
    """
    Spline change in magnetism throughout layer.

    Defines monotonic splines for rhoM and thetaM with shared knot positions.

    *z* is position of the knot in [0, 1] relative to the magnetic layer
    thickness.  The *z* coordinates are automatically sorted before
    rendering, leading to multiple equivalent solutions if knots are swapped.

    *rhoM* gives the magnetic scattering length density for each knot.

    *thetaM* gives the magnetic angle for each knot.

    *name* is the base name for the various layer parameters.

    *dead_above* and *dead_below* define magnetically dead layers at the
    nuclear boundaries.  These can be negative if magnetism extends beyond
    the nuclear boundary.

    *interface_above* and *interface_below* define the magnetic interface
    at the boundaries, if it is different from the nuclear interface.
    """

    z: List[Parameter]
    rhoM: List[Parameter]
    thetaM: List[Parameter]

    magnetic = True

    def __init__(self, z=(), rhoM=(), thetaM=(), name="LAYER", **kw):
        BaseMagnetism.__init__(self, name=name, **kw)

        def parvec(vector, name, limits):
            return [Parameter.default(p, name=name + "[%d]" % i, limits=limits) for i, p in enumerate(vector)]

        self.rhoM, self.thetaM, self.z = [
            parvec(v, name + " " + part, limits)
            for v, part, limits in zip((rhoM, thetaM, z), ("rhoM", "thetaM", "z"), ((None, None), (0, 360), (0, 1)))
        ]
        if len(self.z) != len(self.rhoM):
            raise ValueError("must have one position z for each rhoM")
        if len(self.thetaM) > 0 and len(self.rhoM) != len(self.thetaM):
            raise ValueError("must have one thetaM for each rhoM")

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(rhoM=self.rhoM, thetaM=self.thetaM, z=self.z)
        return parameters

    def to_dict(self):
        result = BaseMagnetism.to_dict(self)
        result["z"] = to_dict(self.z)
        result["rhoM"] = to_dict(self.rhoM)
        result["thetaM"] = to_dict(self.thetaM)
        return result

    def profile(self, Pz, thickness):
        mbelow, tbelow = 0, (self.thetaM[0].value if self.thetaM else DEFAULT_THETA_M)
        mabove, tabove = 0, (self.thetaM[-1].value if self.thetaM else DEFAULT_THETA_M)
        z = np.sort([0.0] + [p.value for p in self.z] + [1.0]) * thickness

        rhoM = np.hstack((mbelow, [p.value for p in self.rhoM], mabove))
        PrhoM = monospline(z, rhoM, Pz)

        if np.any(np.isnan(PrhoM)):
            print("in mono with bad PrhoM")
            print("z %s" % str(z))
            print("p %s" % str([p.value for p in self.z]))

        if len(self.thetaM) > 1:
            thetaM = np.hstack((tbelow, [p.value for p in self.thetaM], tabove))
            PthetaM = monospline(z, thetaM, Pz)
        elif len(self.thetaM) == 1:
            PthetaM = self.thetaM[0].value * np.ones_like(PrhoM)
        else:
            PthetaM = DEFAULT_THETA_M * np.ones_like(PrhoM)
        return PrhoM, PthetaM

    def render(self, probe, slabs, thickness, anchor, sigma):
        Pw, Pz = slabs.microslabs(thickness)
        rhoM, thetaM = self.profile(Pz, thickness)
        slabs.add_magnetism(anchor=anchor, w=Pw, rhoM=rhoM, thetaM=thetaM, sigma=sigma)

    def __str__(self):
        return "freemag(%d)" % (len(self.rhoM))

    def __repr__(self):
        return "FreeMagnetism"


@dataclass(init=False)
class FreeMagnetismInterface(BaseMagnetism):
    r"""
    Spline change in magnetism throughout layer.

    Defines monotonic splines for rhoM and thetaM with shared knot positions.

    *dz* is the relative $z$ step between the knots, with position $z_k$
    defined by $z_k = w \sum_{i=0}^k \delta z_i / \sum_{i=0}^n \delta z_i$,
    where $n$ is the number of intervals. The resulting $z$ must be monotonic,
    with $\delta z_i \ge 0$ for all intervals.

    *drhoM* gives the relative $\rho_M$ step between knots. Unlike
    $\rho_{Mk} = \sum_{i=0}^k \delta \rho_{Mi} / \sum_{i=0}^n \delta \delta \rho_{Mi}$.

    *dthetaM* gives the magnetic angle for each knot.

    *name* is the base name for the various layer parameters.

    *dead_above* and *dead_below* define magnetically dead layers at the
    nuclear boundaries.  These can be negative if magnetism extends beyond
    the nuclear boundary.

    *interface_above* and *interface_below* define the magnetic interface
    at the boundaries, if it is different from the nuclear interface.

    *mbelow* and *mabove* are the rhoM parameter values of
    the above and below layers respectively. Do not specify if
    layers either side are not magnetic.

    *tbelow* and *tabove* are the thetaM parameter values of
    the above and below layers respectively. Do not specify if
    layers either side are not magnetic.
    """

    name: str
    mbelow: Parameter
    mabove: Parameter
    tbelow: Parameter
    tabove: Parameter
    dz: List[Parameter]
    drhoM: List[Parameter]
    dthetaM: List[Parameter]

    magnetic = True

    def __init__(
        self, dz=(), drhoM=(), dthetaM=(), mbelow=0, mabove=0, tbelow=None, tabove=None, name="MagInterface", **kw
    ):
        BaseMagnetism.__init__(self, name=name, **kw)

        def parvec(vector, name, limits):
            return [Parameter.default(p, name=name + "[%d]" % i, limits=limits) for i, p in enumerate(vector)]

        self.drhoM, self.dthetaM, self.dz = [
            parvec(v, name + " " + part, limits)
            for v, part, limits in zip((drhoM, dthetaM, dz), ("drhoM", "dthetaM", "dz"), ((-1, 1), (-1, 1), (0, 1)))
        ]
        self.mbelow = Parameter.default(mbelow, name=name + " mbelow", limits=(-np.inf, np.inf))
        self.mabove = Parameter.default(mabove, name=name + " mabove", limits=(-np.inf, np.inf))

        # if only tabove or tbelow is defined then they are made equal
        # this is to deal with the situation of a non-magnetic
        # layer next to a magnetic one
        if tabove is None and tbelow is None:
            tbelow = tabove = DEFAULT_THETA_M
        elif tbelow is None:
            tbelow = tabove
        elif tabove is None:
            tabove = tbelow

        self.tbelow = Parameter.default(tbelow, name=name + " tbelow", limits=(0, 360))
        self.tabove = Parameter.default(tabove, name=name + " tabove", limits=(0, 360))
        if len(self.dz) != len(self.drhoM):
            raise ValueError("Need one dz for each drhoM")
        if len(self.dthetaM) > 0 and len(self.drhoM) != len(self.dthetaM):
            raise ValueError("Need one dthetaM for each drhoM")

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update(
            drhoM=self.drhoM,
            dthetaM=self.dthetaM,
            dz=self.dz,
            mbelow=self.mbelow,
            mabove=self.mabove,
            tbelow=self.tbelow,
            tabove=self.tabove,
        )
        return parameters

    def profile(self, Pz, thickness):
        z = np.hstack((0, np.cumsum(np.asarray([v.value for v in self.dz], "d"))))
        if z[-1] == 0:
            z[-1] = 1
        z *= thickness / z[-1]

        rhoM_fraction = np.hstack((0, np.cumsum(np.asarray([v.value for v in self.drhoM], "d"))))
        # AJC added since without the line below FreeMagnetismInterface
        # does not initialise properly - fixes strange behaviour at drho=0 on end point
        if rhoM_fraction[-1] == 0:
            rhoM_fraction[-1] = 1

        rhoM_fraction *= 1 / rhoM_fraction[-1]
        PrhoM = np.clip(monospline(z, rhoM_fraction, Pz), 0, 1)

        if self.dthetaM:
            thetaM_fraction = np.hstack((0, np.cumsum(np.asarray([v.value for v in self.dthetaM], "d"))))
            if thetaM_fraction[-1] == 0:
                thetaM_fraction[-1] = 1

            thetaM_fraction *= 1 / thetaM_fraction[-1]
            PthetaM = np.clip(monospline(z, thetaM_fraction, Pz), 0, 1)
        else:
            # AJC changed from len(z) to PrhoM - since PrhoM is the length of the vector
            # we want PthetaM to match - otherwise slabs.add_magnetism throws an error
            PthetaM = np.linspace(0.0, 1.0, len(PrhoM))

        return PrhoM, PthetaM

    def render(self, probe, slabs, thickness, anchor, sigma):
        Pw, Pz = slabs.microslabs(thickness)
        rhoM_profile, thetaM_profile = self.profile(Pz, thickness)
        mbelow, mabove = self.mbelow.value, self.mabove.value
        tbelow, tabove = self.tbelow.value, self.tabove.value
        rhoM = (1 - rhoM_profile) * mbelow + rhoM_profile * mabove
        thetaM = (1 - thetaM_profile) * tbelow + thetaM_profile * tabove
        slabs.add_magnetism(anchor=anchor, w=Pw, rhoM=rhoM, thetaM=thetaM, sigma=sigma)

    def __str__(self):
        return "freemagint(%d)" % (len(self.drhoM))

    def __repr__(self):
        return "FreeMagnetismInterface"
