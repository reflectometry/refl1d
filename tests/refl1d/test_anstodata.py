import os.path

from numpy.testing import assert_almost_equal, assert_equal

from refl1d.anstodata import Platypus

# path to the test file
pth = os.path.dirname(__file__)

PLP = Platypus()
probe = PLP.load(os.path.join(pth, "c_PLP0000708.dat"))

assert_equal(probe.Q.size, 90)
assert_almost_equal(probe.dQ[0], 0.000319677 / 2.3548)
