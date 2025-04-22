# Author: Paul Kienzle
"""
Reflectometry models

Reflectometry models consist of 1-D stacks of layers. Layers are joined
by gaussian interfaces. The layers themselves may be uniform, or the
scattering density may vary with depth in the layer.

.. Note::
   By importing model, the definition of :class:`material.Scatterer <refl1d.sample.material.Scatterer>`
   changes so that materials can be stacked into layers using operator
   overloading:
   - the | operator, (previously known as "bitwise or") joins stacks
   - the * operator repeats stacks (n times, n is an int)

   This will affect all instances of the Scatterer class, and all of its subclasses.
"""

# TODO: xray has smaller beam spot
# => smaller roughness
# => thickness depends on where the beam spot hits the sample
# Xray thickness variance = neutron roughness - xray roughness

__all__ = ["Repeat", "Slab", "Stack", "Layer"]

from copy import copy
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import numpy as np
from bumps.parameter import Calculation, Parameter

from .. import profile
from ..probe.probe import NeutronProbe, XrayProbe
from ..sample import material as mat
from .magnetism import BaseMagnetism


# @dataclass(init=False)
class Layer:  # Abstract base class
    """
    Component of a material description.

    thickness (Parameter: angstrom)
        Thickness of the layer
    interface (Parameter: angstrom)
        Interface for the top of the layer.
    magnetism (Magnetism info)
        Magnetic profile anchored to the layer.
    """

    name: str
    thickness: Parameter
    interface: Optional[Parameter] = None
    magnetism: Optional[BaseMagnetism] = None

    # Trap calls to set magnetism attr so we can update the magnetism parameter
    # names with the layer name when we assign magnetism to the layer
    def __setattr__(self, name, value):
        if name == "magnetism" and value is not None:
            value.set_layer_name(str(self))
        object.__setattr__(self, name, value)

    @property
    def ismagnetic(self):
        return self.magnetism is not None

    def constraints(self):
        """
        Constraints
        """
        return self.thickness >= 0, self.interface >= 0

    def find(self, z):
        """
        Find the layer at depth z.

        Returns layer, start, end
        """
        return self, 0, self.thickness.value

    def parameters(self):
        """
        Returns a dictionary of parameters specific to the layer.  These will
        be added to the dictionary containing interface, thickness and magnetism
        parameters.
        """

    def layer_parameters(self):
        pars = {"thickness": self.thickness}
        if self.interface is not None:
            pars["interface"] = self.interface
        if self.magnetism is not None:
            pars["magnetism"] = self.magnetism.parameters()
        pars.update(self.parameters())
        return pars

    def render(self, probe, slabs):
        """
        Use the probe to render the layer into a microslab representation.
        """

    def penalty(self):
        """
        Return a penalty value associated with the layer.  This should be
        zero if the parameters are valid, and increasing as the parameters
        become more invalid.  For example, if total volume fraction exceeds
        unity, then the penalty would be the amount by which it exceeds
        unity, or if z values must be sorted, then penalty would be the
        amount by which they are unsorted.

        Note that penalties are handled separately from any probability of
        seeing a combination of layer parameters; the final solution to the
        problem should not include any penalized points.
        """
        return 0

    def __str__(self):
        """
        Print shows the layer name
        """
        return getattr(self, "name", repr(self))

    # Define a little algebra for composing samples
    # Layers can be stacked, repeated, or have length/roughness/magnetism set
    def __or__(self, other):
        """Join two layers to make a stack"""
        stack = Stack()
        stack.add(self)
        stack.add(other)
        return stack

    def __mul__(self, other):
        """Repeat a stack or complex layer"""
        if not isinstance(other, int) or not other > 1:
            raise TypeError("Repeat count must be an integer > 1")
        if isinstance(self, Slab):
            raise TypeError("Cannot repeat single slab")
        stack = Stack()
        stack.add(self)
        r = Repeat(stack=stack, repeat=other)
        return r

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, thickness=None, interface=None, magnetism=None):
        c = copy(self)
        # Only set values if they are not None so that defaults
        # carry over from the copied layer
        if thickness is not None:
            c.thickness = Parameter.default(thickness, limits=(0, None), name=self.name + " thickness")
        if interface is not None:
            c.interface = Parameter.default(interface, limits=(0, None), name=self.name + " interface")
        if magnetism is not None:
            c.magnetism = magnetism
        return c


def _parinit(p, v):
    """
    If v is a parameter use v, otherwise use p but with value v.
    """
    if isinstance(v, Parameter):
        p = v
    else:
        p.set(v)
    return p


def _parcopy(p, v):
    """
    If v is a parameter use v, otherwise use a copy of p but with value v.
    """
    if isinstance(v, Parameter):
        p = v
    else:
        p = copy(p)
        p.set(v)
    return p


@dataclass(init=False)
class Slab(Layer):
    """
    A block of material.
    """

    name: str
    thickness: Parameter
    interface: Optional[Parameter] = None
    magnetism: Optional[BaseMagnetism]
    material: Union[mat.SLD, mat.MaterialTypes, mat.Vacuum]

    def __init__(self, material=None, thickness=0, interface=0, name=None, magnetism=None):
        if name is None:
            name = material.name
        self.name = name
        self.material = material
        self.thickness = Parameter.default(thickness, limits=(0, None), name=name + " thickness")
        self.interface = Parameter.default(interface, limits=(0, None), name=name + " interface")
        self.magnetism = magnetism

    def parameters(self):
        return {"material": self.material.parameters()}

    def render(self, probe, slabs):
        rho, irho = self.material.sld(probe)
        w = self.thickness.value
        sigma = self.interface.value
        # print "rho", rho
        # print "irho", irho
        # print "w", w
        # print "sigma", sigma
        slabs.append(rho=rho, irho=irho, w=w, sigma=sigma)

    def __str__(self):
        return self.name
        # return str(self.material)

    def __repr__(self):
        return "Slab(" + repr(self.material) + ")"


@dataclass(init=False)
class Stack(Layer):
    # TODO: Finish this docstring
    """
    Reflectometry layer stack

    A reflectometry sample is defined by a stack of layers. Each layer
    has an interface describing how the top of the layer interacts with
    the bottom of the overlaying layer. The stack may contain
    """

    name: str
    interface: Literal[None]
    layers: List[Union["Slab", "Repeat"]]
    thickness: Parameter = field(metadata={"description": "always equals the sum of the layer thicknesses"})

    def __init__(
        self,
        layers: Optional[Union["Stack", List[Union["Slab", "Repeat"]]]] = None,
        name: str = "Stack",
        interface=None,
        thickness: Optional[Union[Parameter, float]] = None,
    ):
        self.name = name
        self.interface = None
        self._layers = []
        # make sure thickness.id is persistent through serialization:
        thickness = thickness if thickness is not None else 0
        self.thickness = Parameter.default(thickness, name=name + " thickness")
        self._set_thickness()
        if layers is not None:
            self.add(layers)

    @property
    def layers(self):
        return self._layers

    @property
    def ismagnetic(self):
        return any(p.ismagnetic for p in self._layers)

    def find(self, z):
        """
        Find the layer at depth z.

        Returns layer, start, end
        """
        offset = 0
        for L in self._layers:
            dz = L.thickness.value
            if z < offset + dz:
                break
            offset += dz
        else:
            L = self._layers[-1]
            offset -= dz

        L, start, end = L.find(z - offset)
        return L, start + offset, end + offset

    def add(self, other):
        if isinstance(other, Stack):
            self._layers.extend(other._layers)
        elif isinstance(other, Repeat):
            self._layers.append(other)
        else:
            try:
                L = iter(other)
            except TypeError:
                L = [other]
            self._layers.extend(_check_layer(el) for el in L)

    def __setstate__(self, state):
        # this is a temporary shim (2024-10-29), to accomodate objects that were pickled
        # before the custom __getstate__ was removed.
        # TODO: the entire __setstate__ can be removed someday (e.g. in 2026?)
        if isinstance(state, tuple):
            self.interface, self._layers, self.name, self.thickness = state
        else:
            self.__dict__.update(state)

    def __copy__(self):
        stack = Stack()
        stack.interface = self.interface
        stack._layers = self._layers[:]
        stack.thickness = self.thickness
        return stack

    def __len__(self):
        return len(self._layers)

    def __str__(self):
        return " | ".join("%s(%.3g)" % (L, L.thickness.value) for L in self._layers)

    def __repr__(self):
        return "Stack(" + ", ".join(repr(L) for L in self._layers) + ")"

    def parameters(self):
        layers = [L.layer_parameters() for L in self._layers]
        return {"thickness": self.thickness, "layers": layers}

    def penalty(self):
        return sum(L.penalty() for L in self._layers)

    def _calculate_thickness(self):
        return sum(L.thickness.value for L in self._layers)

    def _set_thickness(self):
        desc = """sum of the thickness of the component layers in the stack"""
        calculation = Calculation(description=desc)
        calculation.set_function(self._calculate_thickness)
        self.thickness.slot = calculation

    def render(self, probe, slabs):
        """
        Render the stack into slabs.
        """
        if any(layer.magnetism is not None for layer in self._layers):
            return self._render_magnetic(probe, slabs)
        else:
            return self._render_nonmagnetic(probe, slabs)

    def _render_nonmagnetic(self, probe, slabs):
        """
        Render and sld stack in which no layers are magnetic.
        """
        for layer in self._layers:
            layer.render(probe, slabs)

    def _render_magnetic(self, probe, slabs):
        """
        Render and sld stack in which some layers are magnetic.

        If the magnetism interface above or below is left unspecified, the
        corresponding nuclear interface is used.
        """
        magnetism = None
        end_layer = -1
        for i, layer in enumerate(self._layers):
            # Trigger start of a magnetic layer
            if layer.magnetism is not None:
                if magnetism is not None:
                    raise IndexError("magnetic layer %s overlap" % magnetism)
                magnetism = layer.magnetism
                # import sys; print >>sys.stderr, "magnetism", magnetism
                anchor = slabs.thickness() + magnetism.dead_below.value
                s_below = (
                    np.nan
                    if i == 0
                    else magnetism.interface_below.value
                    if magnetism.interface_below is not None
                    else slabs.surface_sigma
                )
                end_layer = i + magnetism.extent - 1

            # Render nuclear layer
            layer.render(probe, slabs)

            # Wait for end of magnetic layer
            if i == end_layer:
                s_above = (
                    magnetism.interface_above.value if magnetism.interface_above is not None else slabs.surface_sigma
                )
                w = (slabs.thickness() - magnetism.dead_above.value) - anchor
                magnetism.render(probe, slabs, thickness=w, anchor=anchor, sigma=(s_below, s_above))
                magnetism = None

        if magnetism:
            raise IndexError("magnetic layer %s is incomplete" % magnetism)

    def _plot(self, dz=1, roughness_limit=0):
        # TODO: unused?
        import matplotlib.pyplot as plt

        from . import material

        neutron_probe = NeutronProbe(T=np.arange(0, 5, 100), L=5.0)
        xray_probe = XrayProbe(T=np.arange(0, 5, 100), L=1.54)
        slabs = profile.Microslabs(1, dz=dz)

        plt.subplot(211)
        cache = material.ProbeCache(xray_probe)
        slabs.clear()
        self.render(cache, slabs)
        z, rho, irho = slabs.step_profile()
        plt.plot(z, rho, "-g", z, irho, "-b")
        z, rho, irho = slabs.smooth_profile(dz=1, roughness_limit=roughness_limit)
        plt.plot(z, rho, ":g", z, irho, ":b")
        plt.legend(["rho", "irho"])
        plt.xlabel("depth (A)")
        plt.ylabel("SLD (10^6 inv A**2)")
        plt.text(0.05, 0.95, r"Cu-$K_\alpha$ X-ray", va="top", ha="left", transform=plt.gca().transAxes)

        plt.subplot(212)
        cache = material.ProbeCache(neutron_probe)
        slabs.clear()
        self.render(cache, slabs)
        z, rho, irho = slabs.step_profile()
        plt.plot(z, rho, "-g", z, irho, "-b")
        z, rho, irho = slabs.smooth_profile(dz=1, roughness_limit=roughness_limit)
        plt.plot(z, rho, ":g", z, irho, ":b")
        plt.legend(["rho", "irho"])
        plt.xlabel("depth (A)")
        plt.ylabel("SLD (10^6 inv A**2)")
        plt.text(0.05, 0.95, "5 A neutron", va="top", ha="left", transform=plt.gca().transAxes)

    # Stacks as lists
    def _find_by_material(self, target):
        """
        Iterate over all layers that have the given material.
        """
        for i, layer in enumerate(self._layers):
            if hasattr(layer, "stack"):
                for sub in layer.stack._find_by_material(target):
                    yield sub
            elif hasattr(layer, "material"):
                if id(layer.material) == id(target):
                    yield self, i

    def _find_by_name(self, target):
        """
        Iterate over all layers that have the given name.
        """
        for i, layer in enumerate(self._layers):
            if hasattr(layer, "stack"):
                for sub in layer.stack._find_by_name(target):
                    yield sub
            else:
                if str(layer) == target:
                    yield self, i

    def _lookup(self, idx):
        """
        Lookup a layer by integer index, name, material or (material, repeat) if not the first
        occurrence of the material in the sample.  Search is depth first.  Returns the stack
        or substack that contains the material, and the index in that stack.
        """
        if isinstance(idx, int):
            return self, idx

        if isinstance(idx, slice):
            start = (self, 0) if idx.start is None else self._lookup(idx.start)
            stop = (self, len(self)) if idx.stop is None else self._lookup(idx.stop)
            if start[0] != stop[0]:
                raise IndexError("start and and stop of sample slice must be in the same stack")
            return start[0], slice(start[1], stop[1], idx.step)

        # Check for lookup of the nth occurrence of a given layer
        if isinstance(idx, tuple):
            target, count = idx
        else:
            target, count = idx, 1

        # Check if lookup by material or by name
        if isinstance(target, mat.Scatterer):
            sequence = self._find_by_material(target)
        elif isinstance(target, str):
            sequence = self._find_by_name(target)
        else:
            raise TypeError("expected integer, material or layer name as sample index")

        # Move to the nth item in the sequence
        i = -1
        for i, el in enumerate(sequence):
            if i + 1 == count:
                return el
        if i == -1:
            raise IndexError("layer %s not found" % str(target))
        else:
            raise IndexError("only found %d layers of %s" % (str(target), i + 1))

    def __getitem__(self, idx):
        # import sys;print >>sys.stderr, "lookup idx", idx
        stack, idx = self._lookup(idx)
        # print >>sys.stderr, "found", idx
        if isinstance(idx, slice):
            newstack = Stack()
            newstack._layers = stack._layers[idx]
            return newstack
        else:
            return stack._layers[idx]

    def __setitem__(self, idx, other):
        stack, idx = self._lookup(idx)
        if isinstance(idx, slice):
            if isinstance(other, Stack):
                stack._layers[idx] = other._layers
            else:
                stack._layers[idx] = [_check_layer(el) for el in other]
        else:
            stack._layers[idx] = _check_layer(other)

    def __delitem__(self, idx):
        stack, idx = self._lookup(idx)
        # works the same for slices and individual indices
        del stack._layers[idx]

    def insert(self, idx, other):
        """
        Insert structure into a stack.  If the inserted element is
        another stack, the stack will be expanded to accommodate.  You
        cannot make nested stacks.
        """
        stack, idx = self._lookup(idx)
        if isinstance(other, Stack):
            for i, L in enumerate(other._layers):
                stack._layers.insert(idx + i, L)
        elif isinstance(other, Repeat):
            stack._layers.insert(idx, other)
        else:
            try:
                other = iter(other)
            except Exception:
                other = [other]
            for i, L in enumerate(other):
                stack._layers.insert(idx + i, _check_layer(L))

    # Define a little algebra for composing samples
    # Stacks can be repeated or extended
    def __mul__(self, other):
        if isinstance(other, Parameter):
            pass
        elif isinstance(other, int) and other > 1:
            pass
        else:
            raise TypeError("Repeat count must be an integer > 1")
        s = Repeat(stack=self, repeat=other)
        return s

    def __rmul__(self, other):
        return self.__mul__(other)

    def __or__(self, other):
        stack = Stack()
        stack.add(self)
        stack.add(other)
        return stack

    render.__doc__ = Layer.render.__doc__


def _check_layer(el):
    if isinstance(el, Layer):
        return el
    elif isinstance(el, mat.Scatterer):
        return Slab(el)
    else:
        raise TypeError(f"Can only stack materials and layers, not {el} (of type {type(el)})")


@dataclass(init=False)
class Repeat(Layer):
    """
    Repeat a layer or stack.

    If an interface parameter is provide, the roughness between the
    multilayers may be different from the roughness between the repeated
    stack and the following layer.

    Note: Repeat is not a type of Stack, but it does have a stack inside.
    """

    name: str
    interface: Parameter
    magnetism: Optional[BaseMagnetism]
    repeat: Parameter
    stack: Stack
    thickness: Parameter

    def __init__(
        self, stack, repeat=1, interface=None, name=None, magnetism=None, thickness: Optional[Parameter] = None
    ):
        name = "multilayer" if name is None else name
        interface = stack[-1].interface.value if interface is None else interface
        self.magnetism = magnetism
        self.name = name
        self.repeat = Parameter(repeat, limits=(0, None), discrete=True, name=name + " repeats")
        self.stack = stack
        # intentionally ignoring what was in thickness argument...
        # is this a good idea?
        self.thickness = Parameter(name=name + " thickness")
        self.thickness.equals(self.stack.thickness * self.repeat)
        self.interface = Parameter.default(interface, limits=(0, None), name=name + " top interface")

    def __setstate__(self, state):
        # this is a temporary shim (2024-10-29), to accomodate objects that were pickled
        # before the custom __getstate__ was removed.
        # TODO: the entire __setstate__ can be removed someday (e.g. in 2026?)
        if isinstance(state, tuple):
            self.interface, self.repeat, self.name, self.stack = state
        else:
            self.__dict__.update(state)

    def penalty(self):
        return self.stack.penalty()

    @property
    def ismagnetic(self):
        return self.magnetism is not None or self.stack.ismagnetic

    def find(self, z):
        """
        Find the layer at depth z.

        Returns layer, start, end
        """
        n = self.repeat.value
        unit = self.thickness.value
        if z < n * unit:
            offset = int(z / unit) * unit
            L, start, end = self.stack.find(z - offset)
            return L, start + offset, end + offset
        else:
            offset = n * unit
            L, start, end = self.stack.find(unit)
            return L, start + offset, end + offset

    # Stacks as lists
    def __getitem__(self, idx):
        return self.stack[idx]

    def __setitem__(self, idx, other):
        self.stack[idx] = other

    def __delitem__(self, idx):
        del self.stack[idx]

    def parameters(self):
        pars = {
            "stack": self.stack.parameters(),
            "repeat": self.repeat,
            "thickness": self.thickness,
            "interface": self.interface,
        }
        if self.magnetism:
            pars["magnetism"] = self.magnetism.parameters()
        return pars

    def render(self, probe, slabs):
        nr = self.repeat.value
        if nr <= 0:
            return
        mark = len(slabs)
        self.stack.render(probe, slabs)
        slabs.repeat(mark, int(nr), interface=self.interface.value)

    def __str__(self):
        return "(%s)x%d" % (str(self.stack), self.repeat.value)

    def __repr__(self):
        return "Repeat(%s, %d)" % (repr(self.stack), self.repeat.value)


# Extend the material.Scatterer class so that any scatter can be
# implicitly turned into a slab.
def _material_stacker():
    """
    Allow materials to be used in the stack algebra.  Material can be
    called with thickness, interface, magnetism and turned into a slab.
    So instead of::

        sample = Slab(Si, 0, 5) | Slab(Ni, 100, 5) | Slab(air)

    models can use::

        sample = Si(0, 5) | Ni(100, 5) | air

    WARNING: this adds __or__ and __call__ methods to the material.Scatterer
    base class.  This is a nasty thing to do since those who have to debug
    the Scatterer later will not know where to look for these class
    attributes.  On the flip side, changing the base class definition saves
    us from the nastier problem of having to create a sister hierarchy of
    stackable scatterers mirroring the scatterers hierarchy, or the nasty
    problem of a circular definition that Slab depends on Scatterer and
    Scatterer depends on Slab.
    """

    # Note: should have been add these as a Mixin class, but couldn't
    # get it to work on python 3.3
    def __or__(self, other):
        # need __or__ for stacks which start with a bare material, such
        # as Si | air
        stack = Stack()
        # Note: stack.add() converts materials to slabs
        stack.add(self)
        stack.add(other)
        return stack

    def __call__(self, thickness=0, interface=0, magnetism=None):
        slab = Slab(material=self, thickness=thickness, interface=interface, magnetism=magnetism)
        return slab

    mat.Scatterer.__or__ = __or__
    mat.Scatterer.__call__ = __call__


_material_stacker()
