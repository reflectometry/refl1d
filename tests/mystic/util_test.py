# This program is public domain
import sys; sys.path.append('..')
import mystic.util as module

import numpy as np
from math import sqrt

def test():
    v,nv = module.runlength(np.array([1,1,1,2,2,1,3,3]))
    assert np.all(v == [1,2,1,3])
    assert np.all(nv == [3,2,1,2])

    v,nv = module.countunique(np.array([[1,2],[3,2],[1,3],[2,1]]))
    assert np.all(v == [1,2,3])
    assert np.all(nv == [3,3,2])

    n,k,r = 50,5,10000
    stats = np.array([module.choose_without_replacement(n,k)
                         for i in range(r)])
    v,nv = module.countunique(stats)
    assert np.all(v == range(n))
    # Approximately poisson, with mu = k/n*r, sigma = sqrt(mu)
    np.random.seed(1)
    mu = float(k)/float(n)*r
    assert np.all(abs(nv - mu)<3*sqrt(mu)), "try different seed (k small)"

    n,k,r = 10,8,10000
    stats = np.array([module.choose_without_replacement(n,k)
                        for i in range(r)])
    v,nv = module.countunique(stats)
    assert np.all(v == range(n))
    mu = float(k)/float(n)*r
    assert np.all(abs(nv - mu)<3*sqrt(mu)), "try different seed (k large)"


if __name__ == "__main__":
    import doctest
    doctest.testmod(module)
    test()
