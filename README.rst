Refl1D
======

Refl1D is a program for analyzing 1-D reflectometry measurements made with
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

Documentation is available at `<https://refl1d.readthedocs.io>`_. See
`CHANGES.rst <https://github.com/reflectometry/refl1d/blob/master/CHANGES.rst>`_
for details on recent changes. Information on the refl1d release process is in `release notes <release.md>`_.

Use ``pip install refl1d wxpython`` to install in your python environment.

For the windows application, follow the installation instructions on the
`latest release <https://github.com/reflectometry/refl1d/releases/latest>`_
page.  (For the latest bleeding-edge build, see the 
`unstable release <https://github.com/reflectometry/refl1d/releases/tag/sid>`_)

Submit requests and pull requests to the project
`git pages <https://github.com/reflectometry/refl1d>`_

|CI| |RTD| |DOI|

.. |CI| image:: https://github.com/reflectometry/refl1d/workflows/Test/badge.svg
   :alt: Build status
   :target: https://github.com/reflectometry/refl1d/actions

.. |DOI| image:: https://zenodo.org/badge/1757015.svg
   :alt: DOI tag
   :target: https://zenodo.org/doi/10.5281/zenodo.1249715

.. |RTD| image:: https://readthedocs.org/projects/refl1d/badge/?version=latest
   :alt: Documentation status
   :target: https://refl1d.readthedocs.io/en/latest
