# Magic to import a module into the refl1d namespace
import os, imp; imp.load_source('refl1d.flayer',os.path.join(os.path.dirname(__file__),'flayer.py'))


from refl1d.names import *
from numpy import sin, pi, log, exp, hstack

from refl1d.flayer import FunctionalProfile as FP
from refl1d.flayer import FunctionalMagnetism as FM

import numpy
from contextlib import contextmanager
@contextmanager
def seed(n):
    state = numpy.random.get_state()
    numpy.random.seed(n)
    yield
    numpy.random.set_state(state)

sys.dont_write_bytecode = True

def nuc(z, period, phase):
    return sin(2*pi*(z/period + phase))

def mag(z, z1, z2, M1, M2, M3):
    r"""
    Return the following function:

    .. math::

        f(z) = \left{ \begin{array}{ll}
            C & \mbox{if } z < z_1 \\
            re^{kz} & \mbox{if } z_1 \leq z \leq z_2 \\
            az+b & \mbox{if } z > z_2
            \end{array} \right.

    where $C = M_1$, $r,k$ are set such that $re^{kz_1} = M_1$ and
    $re^{kz_2} = M_2$, and $a,b$ are set such that $az_2 + b = M_2$
    and $az_{\rm end} + b = M_3$.
    """
    if z1>z2: z1,z2 = z2,z1
    C = M1
    k = (log(M2) - log(M1)) / (z2-z1)
    r = M1/exp(k*z1)
    a = (M3-M2) / (z[-1] - z2)
    b = M2 - a*z2

    def v1():
        part1 = z[z<z1]*0+C
        part2 = r*exp(k*z[(z>=z1)&(z<=z2)])
        part3 = a*z[z>z2] + b
        return hstack((part1, part2, part3))

    def v2():
        return [ C if zi<z1
                 else r*exp(k*zi) if zi <= z2
                 else a*zi + b
                 for zi in z ]
    def v3():
        ret = []
        for zi in z:
            if zi < z1: ret.append(C)
            elif zi <= z2: ret.append(r*exp(k*zi))
            else: ret.append(a*zi + b)
        return ret

    def v4():
        ret = numpy.empty_like(z)
        part1 = z<z1
        part2 = (z>=z1) & (z<=z2)
        part3 = z > z2
        ret[part1] = C
        ret[part2] = r*exp(k*z[part2])
        ret[part3] = a*z[part3] + b
        return ret
    return v2()

sample = (silicon(0,5)
          | FP(100,0,name="sin",profile=nuc, period=10, phase=0.2,
               magnetism=FM(profile=mag, M1=1, M2=4, M3=5, z1=10, z2=40))
          | air)

sample['sin'].period.range(0,100)
sample['sin'].phase.range(0,1)
sample['sin'].thickness.range(0,1000)
sample['sin'].magnetism.M1.range(0,10)
sample['sin'].magnetism.M2.range(0,10)
sample['sin'].magnetism.M3.range(0,10)
sample['sin'].magnetism.z1.range(0,100)
sample['sin'].magnetism.z2.range(0,100)

T = numpy.linspace(0, 5, 100)
xs = [NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475) for _ in range(4)]
probe = PolarizedNeutronProbe(xs)
M = Experiment(probe=probe, sample=sample, dz=0.1)
with seed(1): M.simulate_data(5)
problem = FitProblem(M)
