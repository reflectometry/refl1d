# This program is public domain
import sys; sys.path.append('..')
import mystic.util as module

import numpy
from math import sqrt

def test():
    v,nv = module.runlength(numpy.array([1,1,1,2,2,1,3,3]))
    assert numpy.all(v == [1,2,1,3])
    assert numpy.all(nv == [3,2,1,2])

    v,nv = module.countunique(numpy.array([[1,2],[3,2],[1,3],[2,1]]))
    assert numpy.all(v == [1,2,3])
    assert numpy.all(nv == [3,3,2])

    n,k,r = 50,5,10000
    stats = numpy.array([module.choose_without_replacement(n,k)
                         for i in range(r)])
    v,nv = module.countunique(stats)
    assert numpy.all(v == range(n))
    # Approximately poisson, with mu = k/n*r, sigma = sqrt(mu)
    numpy.random.seed(1)
    mu = float(k)/float(n)*r
    assert numpy.all(abs(nv - mu)<3*sqrt(mu)), "try different seed (k small)"

    n,k,r = 10,8,10000
    stats = numpy.array([module.choose_without_replacement(n,k)
                         for i in range(r)])
    v,nv = module.countunique(stats)
    assert numpy.all(v == range(n))
    mu = float(k)/float(n)*r
    assert numpy.all(abs(nv - mu)<3*sqrt(mu)), "try different seed (k large)"


if __name__ == "__main__":
    import doctest
    doctest.testmod(module)
    test()
