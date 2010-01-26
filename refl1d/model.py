# This program is in the public domain
# Author: Paul Kienzle
__all__ = ['Bspline','PBS','Slab','Stack','Experiment']

from copy import copy, deepcopy
import numpy
from numpy import (inf, nan, pi, sin, cos, tan, sqrt, exp, log, log10,
                   degrees, radians, floor, ceil)
import periodictable
import periodictable.xsf as xsf
import periodictable.nsf as nsf

from mystic import Parameter as Par, IntegerParameter as IntPar

from .interface import Erf
from . import material

class Layer: # Abstract base class
    """
    Component of a material description.

    thickness (Parameter: angstrom)
        Thickness of the layer
    interface (Interface function)
        Interface for the top of the layer.
    """
    thickness = None
    interface = None
    def constraints(self):
        """
        Constraints
        """
        return self.thickness >= 0, self.roughness >= 0
    def parameters(self):
        """
        Returns a list of parameters used in the layer.
        """
    def render(self, probe, slabs):
        """
        Use the probe to render the layer into a microslab repressentation.
        """

    # Define a little algebra for composing samples
    # Layers can be stacked, repeated, or have length/roughness set
    def __add__(self, other):
        """Join two layers to make a stack"""
        s = Stack()
        s.add(self)
        s.add(other)
        return s
    def __mul__(self, other):
        """Repeat a stack or complex layer"""
        if not isinstance(other, int) or not other > 1:
            raise TypeError("Repeat count must be an integer > 1")
        if isinstance(self, Slab):
            raise TypeError("Cannot repeat single slab""")
        s = Stack()
        s.add(self)
        r = Repeat(stack=s, repeat=other)
        return r
    def __rmul__(self, other):
        return self.__mul__(other)
    def __div__(self, other):
        """Return a new layer with a different thickness"""
        c = copy(self)
        c.thickness = copy(self.thickness)
        c.thickness.set(other)
        return c
    def __idiv__(self, other):
        self.thickness.set(other)
        return self
    def __rdiv__(self, other):
        raise NotImplementedError("Use layer/thickness, not thickness/layer")
    def __mod__(self, other):
        """Return a new layer with a different roughness"""
        c = copy(self)
        c.interface = copy(self.interface)
        c.interface.set(other)
        return c
    def __imod__(self, other):
        self.interface.set(other)
        return self
    def __rmod__(self, other):
        raise NotImplementedError("Use layer%roughness, not roughness%layer")

class Stack(Layer):
    """
    Reflectometry layer stack

    A reflectometry sample is defined by a stack of layers.  Each layer
    has an interface describing how the top of the layer interacts with
    the bottom of the overlaying layer.  The stack may contain

    The sta
    """
    def __init__(self, repeat=1):
        self.interface = None
        self._layers = []
    def add(self, other):
        if isinstance(other,Stack):
            self._layers.extend(other._layers)
        elif isinstance(other,Layer):
            self._layers.append(other)
        elif isinstance(other,material.Scatterer):
            self._layers.append(Slab(other))
        elif isinstance(other, list):
            for el in other:
                if not isinstance(el, Layer):
                    raise TypeError("Can only stack materials and layers")
            self._layers.extend(other)
        else:
            TypeError("Can only stack materials and layers")
    def __str__(self):
        return " + ".join(str(L) for L in self._layers)
    def __repr__(self):
        return "Stack("+", ".join(repr(L) for L in self._layers)+")"
    def parameters(self):
        return [L.parameters() for L in self._layers]
    def _thickness(self):
        """returns the total thickness of the stack"""
        t = 0
        for L in self._layers:
            t += L.thickness
        return t*self.repeat
    thickness = property(_thickness)
    def render(self, probe, slabs):
        for layer in self._layers:
            layer.render(probe, slabs)

    # Stacks as lists
    def __getitem__(self, idx):
        if isinstance(idx,slice):
            s = Stack()
            s._layers = self._layers[idx]
            return s
        else:
            return self._layers[idx]
    def __setitem__(self, idx, other):
        if isinstance(other,Stack):
            self._layers[idx] = other._layers
        elif isinstance(other,Layer):
            self._layers[idx] = other
        elif isinstance(other,material.Scatterer):
            self._layers[idx] = Slab(other)
        elif isinstance(other, list):
            for el in other:
                if not isinstance(el, Layer):
                    raise TypeError("Can only stack materials and layers")
            self._layers[idx] = other
        else:
            raise TypeError("Can only stack materials and layers")
        if isinstance(idx,slice):
            s = Stack()
            s._layers = self._layers[idx]
            return s
        else:
            return self._layers[idx]
    def __delitem__(self, idx):
        del self._layers[idx]

    # Define a little algebra for composing samples
    # Stacks can be repeated or extended
    def __mul__(self, other):
        if not isinstance(other, int) or not other > 1:
            raise TypeError("Repeat count must be an integer > 1")
        s = Repeat(stack=self, repeat=other)
        return s
    def __rmul__(self, other):
        return self.__mul__(other)
    def __add__(self, other):
        s = Stack()
        s.add(self)
        s.add(other)
        return s
    def __radd__(self, other):
        s = Stack()
        s.add(other)
        s.add(self)
    def __iadd__(self, other):
        s.add(other)
        return s
    render.__doc__ = Layer.render.__doc__

class Repeat(Layer):
    """
    Repeat a layer or stack.


    """
    def __init__(self, stack, repeat=1, interface=None):
        self.repeat = IntPar(repeat, limits=(0,inf),
                             name="repeats")
        self.stack = stack
        self.interface = interface
    def parameters(self):
        if interface is not None:
            return dict(stack=self.stack.parameters,
                        repeat=self.repeat,
                        interface=self.interface)
        else:
            return dict(stack=self.stack.parameters,
                        repeat=self.repeat)
    def render(self, probe, slabs):
        # For repeats, may need to control the roughness between the
        # multilayer repeats separately from the roughness after
        # the stack.
        mark = len(slabs)
        nr = self.repeat.value
        if self.interface is None:
            if nr > 0:
                self.stack.render(probe, slabs)
                slabs.repeat(mark, nr)
        else:
            if nr > 1:
                self.stack.render(probe, slabs)
                slabs.repeat(mark, nr-1)
            self.stack.render(probe,slabs)
            slabs.interface(self.interface)
    def __str__(self):
        return "(%s)x%d"%(str(self.stack),self.repeat.value)
    def __repr__(self):
        return "Repeat(%s, %d)"%(repr(self.stack),self.repeat.value)

# Extend the materials scatterer class so that any scatter can be
# implicitly turned into a slab.
class _MaterialStacker:
    """
    Allows materials to be used in a stack algebra, automatically
    turning them into slabs when they are given a thickness (e.g., M/10)
    or roughness (e.g., M%10), or when they are added together
    (e.g., M1 + M2).
    """
    # Define a little algebra for composing samples
    # Layers can be repeated, stacked, or have length/interface set
    def __add__(self, other):
        """Place a slab of material into a layer stack"""
        s = Stack()
        s.add(self)
        s.add(other)
        return s
    def __div__(self, other):
        """Create a slab with the given thickness"""
        c = Slab(material=self)
        c.thickness.set(other)
        return c
    def __rdiv__(self, other):
        raise NotImplementedError("Use layer/thickness, not thickness/layer")
    def __mod__(self, other):
        """Create a slab with the given roughness"""
        c = Slab(material=self)
        c.interface.set(other)
        return c
    def __rmod__(self, other):
        raise NotImplementedError("Use layer%roughness, not roughness%layer")
material.Scatterer.__bases__ += (_MaterialStacker,)

class Slab(Layer):
    """
    A block of material.
    """
    def __init__(self, material=None, thickness=0, interface=0):
        self.material = material
        self.thickness = Par.default(thickness, limits=(0,inf),
                                     name=material.name+" thickness")
        self.interface = Par.default(interface, limits=(0,inf),
                                     name=material.name+" interface")

    def parameters(self):
        return dict(thickness=self.thickness,
                    interface=self.interface,
                    material=self.material.parameters())

    def render(self, probe, slabs):
        rho, irho = self.material.sld(probe)
        w = self.thickness.value
        sigma = self.interface.value
        slabs.extend(rho=[rho], irho=[irho], w=[w], sigma=[sigma])
    def __str__(self):
        if self.thickness.value > 0:
            return "%s/%.3g"%(str(self.material),self.thickness.value)
        else:
            return str(self.material)
    def __repr__(self):
        return "Slab("+repr(self.material)+")"

class Bspline(Layer):
    """
    A freeform section of the sample modeled with B-splines.

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    number of control points.

    The control points are equally spaced in the layers.  Values at
    the ends are flat.
    """
    def __init__(self, thickness=0, rho=[], irho=[], name="BSpline"):
        self.name = name
        self.thickness = Par.default(thickness, limits=(0,inf),
                                   name=name+" thickness")
        self.rho_points = [Par.default(v) for v in rho]
        self.irho_points = [Par.default(v) for v in irho]
    def parameters(self):
        return dict(rho=self.rho_points,
                    irho=self.irho_points,
                    thickness=self.thickness)
    def render(self, probe, slabs):
        raise NotImplementedError
        thickness = self.thickness.value
        rho = [v.value for v in self.rho]
        irho = [v.value for v in self.irho]
        z = slab.steps()

class PBS(Layer):
    """
    A freeform section of the sample modeled with parametric B-splines.

    Each control point uses a pair of parameters (x,y).

    sld (rho) and imaginary sld (irho) can be modeled with a separate
    number of control points.
    """
    def __init__(self, thickness=0, rho=[], irho=[], name="BSpline"):
        self.name = name
        self.thickness = Par.default(thickness, limits=(0,inf),
                                   name=name+" thickness")
        self.rho_points = [(Par.default(x),Par.default(y)) for x,y in rho]
        self.irho_points = [(Par.default(x),Par.default(y)) for x,y in irho]
    def parameters(self):
        return dict(rho=self.rho_points,
                    irho=self.irho_points,
                    thickness=self.thickness)
    def render(self, probe, slabs):
        raise NotImplementedError
        thickness = self.thickness.value
        rho = [v.value for v in self.rho]
        irho = [v.value for v in self.irho]
        z = slab.steps()
