import numpy
from .parameter import function
__all__ = [
    'exp', 'log', 'log10', 'sqrt',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh','cosh','tanh',
    'sum', 'prod'
    ]

sin = function(numpy.sin)
exp = function(numpy.exp)
log = function(numpy.log)
log10 = function(numpy.log10)
sqrt = function(numpy.sqrt)
sin = function(numpy.sin)
cos = function(numpy.cos)
tan = function(numpy.tan)
asin = function(numpy.arcsin)
acos = function(numpy.arccos)
atan = function(numpy.arctan)
sinh = function(numpy.sinh)
cosh = function(numpy.cosh)
tanh = function(numpy.tanh)
atan2 = function(numpy.arctan2)

sum = function(numpy.sum)
prod = function(numpy.prod)
