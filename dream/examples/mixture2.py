#!/usr/bin/env python

"""
Multimodal demonstration using gaussian mixture model.

The model is a mixture model representing the probability density from a
product of gaussians.

Adjust the width of the gaussians, *S*, to see how the relative diameter of
the modes affects the number of generations required for good sampling.

Note that dream.diffev.de_step adds jitter to the parameters at the 1e-6 level, 
so S < 1e-4 cannot be modeled reliably.
"""
from pylab import *
from dream import *

S = 0.001
model = Mixture(MVNormal([-4, 2],S*eye(2)), 5, 
                MVNormal([-2,-2],S*eye(2)), 2.5,
                MVNormal([ 0,-4],S*eye(2)), 1,
                MVNormal([ 2, 0],S*eye(2)), 4,
                MVNormal([ 4, 4],S*eye(2)), 1,
                )

sampler = Dream(model=model, population=randn(20,2),
                #use_delayed_rejection=False,
                outlier_test='none',
                thinning=1, generations=1000,
                cycles=4)
state = sampler.sample()
plot_all(state)
