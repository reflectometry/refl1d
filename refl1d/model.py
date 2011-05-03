# This program is in the public domain
# Author: Paul Kienzle
"""
Reflectometry models

Reflectometry models consist of 1-D stacks of layers. Layers are joined
by gaussian interfaces. The layers themselves may be uniform, or the
scattering density may vary with depth in the layer.

.. Note::
   By importing model, the definition of :class:`material.Scatterer <refl1d.material.Scatterer>`
   changes so that materials can be stacked into layers using operator
   overloading. This will affect all instances of the Scatterer class,
   and all of its subclasses.
"""

#TODO: xray has smaller beam spot
# => smaller roughness
# => thickness depends on where the beam spot hits the sample
# Xray thickness variance = neutron roughness - xray roughness


__all__ = ['Repeat','Slab','Stack','Layer']

from copy import copy, deepcopy
import numpy
from numpy import (inf, nan, pi, sin, cos, tan, sqrt, exp, log, log10,
                   degrees, radians, floor, ceil)
import periodictable
import periodictable.xsf as xsf
import periodictable.nsf as nsf

from mystic.parameter import Parameter as Par, IntegerParameter as IntPar, Function

from .interface import Erf
from . import material

class Layer(object): # Abstract base class
    """
    Component of a material description.

    thickness (Parameter: angstrom)
        Thickness of the layer
    interface (Interface function)
        Interface for the top of the layer.
    """
    thickness = None
    interface = None
    magnetic = False
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
        Returns a list of parameters used in the layer.
        """
    def render(self, probe, slabs):
        """
        Use the probe to render the layer into a microslab representation.
        """

    def __str__(self):
        """
        Print shows the layer name
        """
        return getattr(self,'name',repr(self))


    # Define a little algebra for composing samples
    # Layers can be stacked, repeated, or have length/roughness set
    def __or__(self, other):
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
    def __call__(self, thickness=None, interface=None):
        c = copy(self)
        if thickness != None:
            c.thickness = Par.default(thickness, limits=(0,inf),
                                      name=self.name+" thickness")
        if interface != None:
            c.interface = Par.default(interface, limits=(0,inf),
                                      name=self.name+" interface")
        return c

def _parinit(p, v):
    """
    If v is a parameter use v, otherwise use p but with value v.
    """
    if isinstance(v, Par):
        p = v
    else:
        p.set(v)
    return p
def _parcopy(p, v):
    """
    If v is a parameter use v, otherwise use a copy of p but with value v.
    """
    if isinstance(v, Par):
        p = v
    else:
        p = copy(p)
        p.set(v)
    return p


class Stack(Layer):
    """
    Reflectometry layer stack

    A reflectometry sample is defined by a stack of layers. Each layer
    has an interface describing how the top of the layer interacts with
    the bottom of the overlaying layer. The stack may contain
    """
    def __init__(self, base=None, name="Stack"):
        self.name = name
        self.interface = None
        self._layers = []
        if base is not None:
            self.add(base)
        # TODO: can we make this a class variable?

        self._thickness = Function(self._calc_thickness,name="stack thickness")

    @property
    def magnetic(self):
        return any(p.magnetic for p in self._layers)
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

        L, start, end = L.find(z-offset)
        return L, start+offset, end+offset

    def add(self, other):
        if isinstance(other,Stack):
            self._layers.extend(other._layers)
        elif isinstance(other,Repeat):
            self._layers.append(other)
        else:
            try:
                L = iter(other)
            except:
                L = [other]
            self._layers.extend(_check_layer(el) for el in L)

    def __getstate__(self):
        return self.interface, self._layers, self.name
    def __setstate__(self, state):
        self.interface, self._layers, self.name = state
        self._thickness = Function(self._calc_thickness,name="stack thickness")
    def __copy__(self):
        newone = Stack()
        newone.interface = self.interface
        newone._layers = self._layers[:]
        return newone
    def __len__(self):
        return len(self._layers)
    def __str__(self):
        return " | ".join("%s(%.3g)"%(L,L.thickness.value)
                          for L in self._layers)
    def __repr__(self):
        return "Stack("+", ".join(repr(L) for L in self._layers)+")"
    def parameters(self):
        layers=[L.parameters() for L in self._layers]

        return dict(thickness=self.thickness, layers = layers)

        #attrs = dict(thickness=self.thickness)
        #return (attrs,layers)
        #return [L.parameters() for L in self._layers]
    def _calc_thickness(self):
        """returns the total thickness of the stack"""
        t = 0
        for L in self._layers:
            t += L.thickness.value
        return t
    @property
    def thickness(self): return self._thickness
    def render(self, probe, slabs):
        for layer in self._layers:
            layer.render(probe, slabs)

    def _plot(self, dz=1, roughness_limit=0):
        import pylab
        import profile, material, probe
        neutron_probe = probe.NeutronProbe(T=numpy.arange(0,5,100), L=5.)
        xray_probe = probe.XrayProbe(T=numpy.arange(0,5,100), L=1.54)
        slabs = profile.Microslabs(1, dz=dz)

        pylab.subplot(211)
        cache = material.ProbeCache(xray_probe)
        slabs.clear()
        self.render(cache, slabs)
        z,rho,irho = slabs.step_profile()
        pylab.plot(z,rho,'-g',z,irho,'-b')
        z,rho,irho = slabs.smooth_profile(dz=1, roughness_limit=roughness_limit)
        pylab.plot(z,rho,':g',z,irho,':b', hold=True)
        pylab.legend(['rho','irho'])
        pylab.xlabel('depth (A)')
        pylab.ylabel('SLD (10^6 inv A**2)')
        pylab.text(0.05,0.95,r"Cu-$K_\alpha$ X-ray", va="top",ha="left",
                   transform=pylab.gca().transAxes)

        pylab.subplot(212)
        cache = material.ProbeCache(neutron_probe)
        slabs.clear()
        self.render(cache, slabs)
        z,rho,irho = slabs.step_profile()
        pylab.plot(z,rho,'-g',z,irho,'-b')
        z,rho,irho = slabs.smooth_profile(dz=1, roughness_limit=roughness_limit)
        pylab.plot(z,rho,':g',z,irho,':b', hold=True)
        pylab.legend(['rho','irho'])
        pylab.xlabel('depth (A)')
        pylab.ylabel('SLD (10^6 inv A**2)')
        pylab.text(0.05,0.95,"5 A neutron", va="top",ha="left",
                   transform=pylab.gca().transAxes)


    # Stacks as lists
    def __getitem__(self, idx):
        if isinstance(idx,slice):
            s = Stack()
            s._layers = self._layers[idx]
            return s
        else:
            return self._layers[idx]
    def __setitem__(self, idx, other):
        if isinstance(idx, slice):
            if isinstance(other,Stack):
                self._layers[idx] = other._layers
            else:
                self._layers[idx] = [_check_layer(el) for el in other]
        else:
            self._layers[idx] = _check_layer(other)
    def __delitem__(self, idx):
        # works the same for slices and individual indices
        del self._layers[idx]

    def insert(self, idx, other):
        """
        Insert structure into a stack.  If the inserted element is
        another stack, the stack will be expanded to accommodate.  You
        cannot make nested stacks.
        """
        if isinstance(other,Stack):
            for i,L in enumerate(other._layers):
                self._layers.insert(idx+i,L)
        elif isinstance(other,Repeat):
            self._layers.insert(idx, other)
        else:
            try:
                other = iter(other)
            except:
                other = [other]
            for i,L in enumerate(other):
                self._layers.insert(idx+i,_check_layer(L))

    # Define a little algebra for composing samples
    # Stacks can be repeated or extended
    def __mul__(self, other):
        if isinstance(other, Par): pass
        elif isinstance(other, int) and other > 1: pass
        else: raise TypeError("Repeat count must be an integer > 1")
        s = Repeat(stack=self, repeat=other)
        return s
    def __rmul__(self, other):
        return self.__mul__(other)
    def __or__(self, other):
        s = Stack()
        s.add(self)
        s.add(other)
        return s

    render.__doc__ = Layer.render.__doc__

def _check_layer(el):
    if isinstance(el,Layer):
        return el
    elif isinstance(el, material.Scatterer):
        return Slab(el)
    else:
        raise TypeError("Can only stack materials and layers, not %s"%el)

class Repeat(Layer):
    """
    Repeat a layer or stack.

    If an interface parameter is provide, the roughness between the
    multilayers may be different from the roughness between the repeated
    stack and the following layer.

    Note: Repeat is not a type of Stack, but it does have a stack inside.
    """
    def __init__(self, stack, repeat=1, interface=None, name=None):
        if name is None: name = "multilayer"
        if interface is None: interface = stack[-1].interface.value
        self.name = name
        self.repeat = IntPar(repeat, limits=(0,inf),
                             name=name + " repeats")
        self.stack = stack
        self.interface = Par.default(interface, limits=(0,inf),
                                     name=name+" top interface")
        # Thickness is computed; don't make it a simple attribute
        self._thickness = Function(self._calc_thickness,name="repeat thickness")
    def __getstate__(self):
        return self.interface, self.repeat, self.name, self.stack
    def __setstate__(self, state):
        self.interface, self.repeat, self.name, self.stack = state
        self._thickness = Function(self._calc_thickness,name="repeat thickness")
    @property
    def magnetic(self):
        return self.stack.magnetic
    def find(self, z):
        """
        Find the layer at depth z.

        Returns layer, start, end
        """
        n = self.repeat.value
        unit = self.thickness.value
        if z < n*unit:
            offset = int(z/unit)*unit
            L,start,end = self.stack.find(z-offset)
            return L,start+offset,end+offset
        else:
            offset = n*unit
            L,start,end = self.stack.find(unit)
            return L,start+offset,end+offset
    # Stacks as lists
    def __getitem__(self, idx):
        return self.stack[idx]
    def __setitem__(self, idx, other):
        self.stack[idx] = other
    def __delitem__(self, idx):
        del self.stack[idx]

    def parameters(self):
        return dict(stack=self.stack.parameters(),
                    repeat=self.repeat,
                    thickness=self._thickness,
                    interface=self.interface)
    # Mark thickness as read only
    @property
    def thickness(self): return self._thickness
    def _calc_thickness(self):
        return self.stack.thickness.value*self.repeat.value
    def render(self, probe, slabs):
        nr = self.repeat.value
        if nr <= 0: return
        mark = len(slabs)
        self.stack.render(probe, slabs)
        slabs.repeat(mark, nr, interface=self.interface.value)
    def __str__(self):
        return "(%s)x%d"%(str(self.stack),self.repeat.value)
    def __repr__(self):
        return "Repeat(%s, %d)"%(repr(self.stack),self.repeat.value)

# Extend the materials scatterer class so that any scatter can be
# implicitly turned into a slab.  This is a nasty thing to do
# since those who have to debug the system later will not know
# to look elsewhere for the class attributes.  On the flip side,
# changing the base class definition saves us the equally nasty
# problem of having to create a sister hierarchy of stackable
# scatterers mirroring the structure of the materials class.
class _MaterialStacker:
    """
    Allows materials to be used in a stack algebra, automatically
    turning them into slabs when they are given a thickness (e.g., M/10)
    or roughness (e.g., M%10), or when they are added together
    (e.g., M1 + M2).
    """
    # Define a little algebra for composing samples
    # Layers can be repeated, stacked, or have length/interface set
    def __or__(self, other):
        """Place a slab of material into a layer stack"""
        s = Stack()
        s.add(self)
        s.add(other)
        return s
    def __call__(self, thickness=0,interface=0):
        c = Slab(material=self, thickness=thickness, interface=interface)
        return c
material.Scatterer.__bases__ += (_MaterialStacker,)

class Slab(Layer):
    """
    A block of material.
    """
    def __init__(self, material=None, thickness=0, interface=0, name=None):
        if name is None: name = material.name
        self.name = name
        self.material = material
        self.thickness = Par.default(thickness, limits=(0,inf),
                                     name=name+" thickness")
        self.interface = Par.default(interface, limits=(0,inf),
                                     name=name+" interface")

    def parameters(self):
        return dict(thickness=self.thickness,
                    interface=self.interface,
                    material=self.material.parameters())

    def render(self, probe, slabs):
        rho, irho = self.material.sld(probe)
        w = self.thickness.value
        sigma = self.interface.value
        #print "rho",rho
        #print "irho",irho
        #print "w",w
        #print "sigma",sigma
        slabs.append(rho=rho, irho=irho, w=w, sigma=sigma)
    def __str__(self):
        if self.thickness.value > 0:
            return str(self.material)
        else:
            return str(self.material)
    def __repr__(self):
        return "Slab("+repr(self.material)+")"
