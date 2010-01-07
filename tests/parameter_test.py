# This program is public domain
import sys; sys.path.append('..')
import math
import mystic.parameter
from mystic.parameter import Parameter as Par
from mystic.pmath import *

def test():
    x = Par(3, name="x")
    y = Par.pm(4,5, name="y")
    z = Par.pmp(2,1, name="z")
    f = 3*x**2 + 2*y + z

    assert x.value == 3
    assert y.value == 4
    assert z.value == 2
    assert (x+y).value == 7
    assert (x*y).value == 12
    assert (x-y).value == -1
    assert float(f) == 3*3**2 + 2*4 + 2
    assert sin(x).value == math.sin(x.value)
    assert float(sum([x,y,z])) == 3+4+2
    assert f == 3*3**2 + 2*4 + 2
    assert bool(x > 2)
    assert x <= 3
    assert x != 4
    assert bool(x < 3) == False

if __name__ == "__main__":
    import doctest
    doctest.testmod(mystic.parameter)
    test()
