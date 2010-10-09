#!/usr/bin/env python

"""
Bevington & Robinson's model of dual exponential decay

References::
    [5] Bevington & Robinson (1992).
    Data Reduction and Error Analysis for the Physical Sciences,
    Second Edition, McGraw-Hill, Inc., New York.
"""

import numpy
from numpy import exp, sqrt
from mystic import Model

def dual_exponential(t,a):
    """
    Computes dual exponential decay.

        y = a1 + a2 exp(-t/a3) + a4 exp(-t/a5)
    """
    a1,a2,a3,a4,a5 = a
    t = numpy.asarray(t)
    return a1 + a2*exp(-t/a4) + a3*exp(-t/a5)

# data from Chapter 8 of [5].
data = numpy.array([[15, 775], [30, 479], [45, 380], [60, 302],
[75, 185], [90, 157], [105,137], [120, 119], [135, 110],
[150, 89], [165, 74], [180, 61], [195, 66], [210, 68],
[225, 48], [240, 54], [255, 51], [270, 46], [285, 55],
[300, 29], [315, 28], [330, 37], [345, 49], [360, 26],
[375, 35], [390, 29], [405, 31], [420, 24], [435, 25],
[450, 35], [465, 24], [480, 30], [495, 26], [510, 28],
[525, 21], [540, 18], [555, 20], [570, 27], [585, 17],
[600, 17], [615, 14], [630, 17], [645, 24], [660, 11],
[675, 22], [690, 17], [705, 12], [720, 10], [735, 13],
[750, 16], [765, 9], [780, 9], [795, 14], [810, 21],
[825, 17], [840, 13], [855, 12], [870, 18], [885, 10]])

f = dual_exponential
x = data[0]
y = data[1]
dy = sqrt(data[1])
del data

class Model(mystic.Model):
    """
    Computes dual exponential decay.

        y = A exp(-t/a) + B exp(-t/b) + C
    """
    def __init__(self, A=0, B=0, C=0, a=1, b=1):
        self._pars = C,A,a,B,b
    def parameters(self):
        return [p for p in self._pars if isinstance(p,Parameter) and p.fitted]
    def __call__(self, t):
        p = [float(v) for v in self._pars]
        return dual_exponential(t,p)

m = Model()
e = Measurement(x,y,dy)


class Measurement(mystic.Experiment):
    def __init__(self):
        pass


class BevingtonDecay(AbstractModel):

    def __init__(self,name='decay',metric=lambda x: numpysum(x*x)):
        AbstractModel.__init__(self,name,metric)
        return

    def evaluate(self,coeffs,evalpts):
        """
        evaluate dual exponential decay with given coeffs over given evalpts
        coeffs = (a1,a2,a3,a4,a5)
        """

    def ForwardFactory(self,coeffs):
        """generates a dual decay model instance from a list of coefficients"""
        a1,a2,a3,a4,a5 = coeffs
        def forward_decay(evalpts):
            """a dual exponential decay over a 1D numpy array
with (a1,a2,a3,a4,a5) = (%s,%s,%s,%s,%s)""" % (a1,a2,a3,a4,a5)
            return self.evaluate((a1,a2,a3,a4,a5),evalpts)
        return forward_decay

    def CostFactory(self,target,pts):
        """generates a cost function instance from list of coefficients & evaluation points"""
        datapts = self.evaluate(target,pts)
        F = CF()
        F.addModel(self.ForwardFactory,self.__name__,len(target))
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=sqrt(datapts),metric=self.__metric__)
        return self.__cost__

    def CostFactory2(self,pts,datapts,nparams):
        """generates a cost function instance from datapoints & evaluation points"""
        F = CF()
        F.addModel(self.ForwardFactory,self.__name__,nparams)
        self.__cost__ = F.getCostFunction(evalpts=pts,observations=datapts,sigma=sqrt(datapts),metric=self.__metric__)
        return self.__cost__

    pass


# prepared instances
decay = BevingtonDecay() #FIXME: look up the correct name for the model!

# cost function with br8.data as the target
cost = decay.CostFactory2(data[:,0],data[:,1],5)


# End of file
