# This program is public domain
import sys; sys.path.append('..')
import mystic.optimizer.diffev_compat as module
from mystic.optimizer.diffev_compat import diffev
from mystic.plotter import plot_response_surface
import pylab

def test():
    f = lambda x: (x[0]-3)**2+(x[1]-5)**2
    point = diffev(f, x0=[2,4], ftol=1e-5)
    print("result %s"%str(point))
    plot_response_surface(f, point, dims=[0,1]); pylab.show()
    assert abs(point[0]-3)+abs(point[1]-5) < 1e-5

if __name__ == "__main__":
    import doctest
    doctest.testmod(module)
    test()
