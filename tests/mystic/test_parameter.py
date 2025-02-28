# This program is public domain
import sys

sys.path.append("..")
import math
import mystic.parameter
from mystic.parameter import Parameter
from mystic.pmath import *


def test():
    x = Parameter(3, name="x")
    y = Parameter(4, name="y").pm(5)
    z = Parameter(2, name="z").pmp(1)
    f = 3 * x**2 + 2 * y + z

    assert x.fixed and not y.fixed and not z.fixed and f.fixed
    assert x.fittable and y.fittable and z.fittable and not f.fittable

    assert x.value == 3
    assert y.value == 4
    assert z.value == 2
    assert (x + y).value == 7
    assert (x * y).value == 12
    assert (x - y).value == -1
    assert float(f) == 3 * 3**2 + 2 * 4 + 2
    assert sin(x).value == math.sin(x.value)
    assert float(sum([x, y, z])) == 3 + 4 + 2
    assert f.value == 3 * 3**2 + 2 * 4 + 2
    assert bool(x > 2)
    assert x <= 3
    assert x != 4
    assert not bool(x < 3)
    # print "condition",x <= 3,"value",bool(x<=3)

    class A(object):
        def __init__(self, b):
            self.b = b

        def call(self):
            return self.b

        def __call__(self):
            return self.b

    obj = A(5)
    p = mystic.parameter.Reference(obj, "b")
    assert p.value == 5
    p.value = 6
    assert obj.b == 6
    assert obj() == 6

    assert p.fittable and p.fixed
    p.pm(3)
    assert not p.fixed

    # Want an automatic wrapper for simple python models such as:
    # class WrappedA(A,mystic.parameter.Parameterized):
    #    __parameters__ = ["b"]

    # This is tricky because for the wrapped class, self.b corresponds to
    # b.value but for the wrapper class, self.b corresponds to the actual
    # parameter which can be replaced or otherwise operated on.  We
    # certainly don't want to rewrite all methods in the wrapped class
    # to used self.b.value rather than self.b, nor do we want to have a
    # different interface for wrapped classes and 'natural' parameterized
    # classes.  Metaclass magic might be able to save us, if it can
    # automatically generate the following from __new__:
    class InternalA(A):
        # Internal class has properties for each parameter
        def __get_b(self):
            return self.__wrapper.b.value

        def __set_b(self, value):
            self.__wrapper.b.value = value

        b = property(__get_b, __set_b)

        # Initialization may refer to parameters by property
        def __init__(self, __wrapper, *args, **kw):
            self.__wrapper = __wrapper
            A.__init__(self, *args, **kw)

    class WrappedA(object):
        __parameters__ = ["b"]

        # Construct parameters and call the internal class constructor
        def __init__(self, *args, **kw):
            for k in self.__parameters__:
                object.__setattr__(self, k, Parameter(name=k))
            a = InternalA(self, *args, **kw)
            object.__setattr__(self, "_internal_class", a)

        # All parameterized objects have a set of parameters
        def parameters(self):
            return dict((k, getattr(self, k)) for k in self.__parameters__)

        # Delegate remaining attributes to the internal object
        def __getattr__(self, a):
            return getattr(self._internal_class, a)

        # Set attributes directly into the class, unless they are parameter
        # objects, in which case they should be set normally.
        def __setattr__(self, a, v):
            if a in self.__parameters__:
                if not isinstance(v, mystic.parameter.BaseParameter):
                    raise TypeError("Attribute '%s' must be a parameter" % a)
                object.__setattr__(self, a, v)
            else:
                setattr(self._internal_class, a, v)

        # Need to explicity forward magic names, except __init__,
        # __getattr__ and __setattr__.
        __call__ = lambda self, *args, **kw: self._internal_class.__call__(*args, **kw)

    # Need to explicity forward magic names
    # setattr(WrappedA,'__call__',
    #        lambda self, *args, **kw: getattr(self._A, '__call__')(*args, **kw))

    # Check that the behaviour works as desired
    wobj = WrappedA(5)
    assert wobj.parameters() == {"b": wobj.b}
    assert wobj.b.fixed and wobj.b.fittable
    assert wobj.call() == 5
    assert wobj() == 5
    wobj.b.pm(3)
    assert not wobj.b.fixed
    wobj.b = f
    assert wobj.parameters() == {"b": f}
    assert wobj.b.fixed and not wobj.b.fittable
    assert wobj() == f.value

    # More complex models with hierarchical structure are more complicated,
    # and would require wrappers for the leaves as well as the containters.
    # E.g.:
    class GaussPeak:
        def __init__(self, sigma=1, mu=0, A=1):
            self.sigma, self.mu, self.A = sigma, mu, A

        def __call__(self, x):
            return A * exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    class PeakList(list):
        def __call__(self, x):
            y = 0
            for p in self:
                y += p(x)
            return y

    # class WrappedGaussPeak(GaussPeak,mystic.parameter.Parameterized):
    #    __parameters__ = ["sigma","mu","A"]
    # class WrappedPeakList(PeakList,mystic.parameter.ParameterizedContainer):
    #    __parameters__ = []
    #    def parameters(self):
    #        return [p.parameters() for p in self.peaks]
    #
    # If parameters were not supplied, the PeakSet wrapper could do
    # a deepcopy traversal of the object, keeping track of the paths to
    # the individual attributes that happen to be BaseParameter instances.
    # It wouldn't be very efficient, but there is value in being automated.
    # Besides which, parameters() is only called once per fit, so efficiency
    # isn't that important.

    # Somewhat more interesting to the third party developer is to take an
    # already existing structure and wrap it for fitting.  By defining a
    # mapping between classes and fittable attributes this process can be
    # automated.  Again, walk the structure checking for classes that
    # match and create parameters for each.
    # mapping = Fittable()
    # mapping.add(cls=GaussPeak, attr=['sigma','mu','A'], update=None)
    # model = PeakList([GaussPeak(sigma=1,mu=2,A=3),GaussPeak()])
    # fit = mapping.scan(model)
    # [{'sigma': 1, 'mu': 2, 'A': 3}, {'sigma': 1, 'mu': 0, 'A': 1}]
    # fit[0].sigma.value = 3
    # The resulting structure contains parameters whose values can be
    # modified, and whose range can be set.  Each of these values will
    # be an alias into the corresponding object.
    # Need to handle parameter expressions.  Here are a couple of options:
    # fit[0].sigma = 2*fit[1].sigma
    # fit[0].sigma.value = 2*fit[1].sigma
    # Both are challenging.


if __name__ == "__main__":
    import doctest

    doctest.testmod(mystic.parameter)
    test()
