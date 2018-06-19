Refl1D
======

Refl1D is a program for analyzing 1D reflectometry measurements made with
X-ray and neutron beamlines.  The 1-D models give the depth profile for
material scattering density composed of a mixture of flat and continuously
varying freeform layers. With polarized neutron measurements, scientists
can study the sub-surface structure of magnetic samples. The architecture
supports the addition of specialized layer types such as models for the
density distribution of polymer brushes, and volume space modeling for
proteins in bio-membranes. We provide a number of these models as well as
supporting user defined layer types for both structural and magnetic
scattering densities.

Fitting is provided by Bumps, a bayesian uncertainty analysis program.  In
addition to the usual uncertain estimated from the covariance at the best
fit location, Bumps includes a Markov chain Monte Carlo analysis code which
more completely describes the uncertain and correlations between parameters.
Fitting is done in parallel, either using python multiprocessing on a
multicore machine, or using MPI for running on a cluster.

See `<https://github.com/reflectometry/refl1d/blob/master/CHANGES.rst>`_ for
details on recent changes.

.. image:: https://travis-ci.org/reflectometry/refl1d.svg?branch=master
    :target: https://travis-ci.org/reflectometry/refl1d

.. image:: https://ci.appveyor.com/api/projects/status/55ps40bauoqw2q6m?svg=true
    :target: https://ci.appveyor.com/project/reflectometry/refl1d
