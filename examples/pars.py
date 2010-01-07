"""
Requirements:

1. Hard limits on parameter values
2. Formulas relating parameter values
3. Likelihoods associated with parameter values
4. Users can turn on or off fitting for individual parameters
5. Model search space requires the ability to add and remove
   parameters from the fit
6. Constraints on calculated parameter values

Desirable characteristics:

1. Wrap pre-existing models

Options:

1. Models only contain values; metadata for fittable parameters is part of an
   adaptor.
2. Models have values and metadata; automate adaptor
3.  and have metadata describing the properties
2. Models contain parameter slots, with shared parameters copied between
   the slots
2. Models contain parameter objects specific to the model, and require
   formulas to relate them to other parameters.
3.

"""


# Fittable mixin
class Fittable(Fittable):
    def __init__(self):
        # walk the attributes of scale, finding any which are fittable.
        pass

# Want to be able to define a model without regard to how it is going to
# be used
class Scale:
    def __init__(self, a):
        self.a = a
    def __call__(self, x):
        return a*x


# Want to be able to parameterize a pre-existing model
class FittableScale(Fittable,Scale):
    parameters=[Parameter('a',limits=(0,1))]
    def __init__(self, *args, **kw):
        Scale.__init__(self, *args, **kw)
        Fittable.__init__(self)

# Actual usage
s = FittableScale(a=5)




fit = Fit(s)
s.a
Fit(s).a.pm(10)
