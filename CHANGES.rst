**************
Change History
**************

Note: updated docs now found at `<http://refl1d.readthedocs.org>`_

2020-02-18 v0.8.10
==================
* add `--checkpoint=n` to save mcmc state every *n* hours.
* hide resolution bars using *Probe.show_resolution = False* in your model.py
* fix residuals, fresnel, logfresnel, and q4 plots when interpolation is used
* fix error contour save function: rhoM was not being written
* fix simulation error: theory and simulated data were sharing the same vector
* fix doc generation: now builds with sphinx 2.4.0 and matplotlib 3.1.0.
* python 3.8 support requires bumps v0.7.14 (released 2020-01-03)

**BREAKING CHANGE**: old-style data loader with sample_broadening set
* sample_broadening was applied twice: once to the base Δθ from when the probe
  was created and again whenever ΔQ was computed; this is not a problem with
  the new *load4* function.  The code was fixed, which may cause difficulties
  when reloading old fits.

2019-10-15 v0.8.9
=================
* json save: material and magnetism are now json objects
* json save: include non-parameter data in save file
* load4: accept 2 and 3 column data
* load4: override resolution and uncertainty given (or missing) from file
* load4: accept multi-entry data as Q probe without knowing theta/lambda
* load4: set the data slice to load
* load4: set default radiation to 'neutron'
* allow fit by number density for materials
* fix interpolation when plotting reflectivity between measured data points
* fix bug in dQ when sample_broadening is initialized to non-zeros
* revised installer: embedded python in zip file
* functions to compute transmission and reflection at each layer
* allow simulation with uncertainty from data
* force minimum uncertainty in data to 1e-11
* change default data view from Fresnel to log10
* apply resolution to saved Fresnel curve
* improved python 3 support

2019-03-01 v0.8.8
=================
* fix json save for MixedExperiments
* save smooth magnetic profiles
* fix abeles code to choose correct branch cut below critical edge
* force absorption to be 0 or positive

2018-12-18 v0.8.7
=================
* make sample broadening a fittable parameter
* allow model + data to be loaded from zip file (bumps 0.7.12 and up)
* improve serialization support

2018-09-24 v0.8.6
=================
* added serialization support
* added option to supply uncertainties when simulating data

2018-06-18 v0.8.5
=================
* fix for plotting spin asymmetry when data is not present (model-only)
* added requirements to setup.py so that `pip install refl1d` suffices

2018-06-14 v0.8.4
=================
* full support for python 3 in GUI
* allow :code:`--pars=parfile` with extra or missing parameters

2018-06-08 v0.8.3
=================
* fix saved magnetic profiles

2018-05-18 v0.8.2
=================
* include new entry points: run program by typing :code:`refl1d` at prompt

2018-05-17 v0.8.1
=================
* allow alternate column order, such as :code:`load4(..., columns="Q dQ R dR")`
* include resolution effects in Fresnel reflectivity normalization
* allow magnetic profile to be saved

2017-12-01 v0.8
===============

* incoherent cross sections now calculated as total minus coherent
* make sure displayed chisq is consistent with negative log likelihood
* allow blending across multiple interfaces
* allow Nevot-Croce calculations for magnetic models

2016-08-05 v0.7.9a2
===================

* support magnetic substrate

2016-08-05 v0.7.8
=================

* load 4-column data: Q, R, dR, dQ, with dQ using 1-sigma resolution
* support Zeeman/Felcher effect for spin-flip in large applied fields
* fix Fresnel calculation
* add --view option from command line to select plot format

2014-11-05 R0.7.7
=================

* add end-tethered and mushroom models for polymers
* support magnetic incident and substrate media
* support Microsoft Visual C compiler
* allow stop after a maximum amount of time (useful in batch queues)
* add entropy calculator

2014-05-30 R0.7.6
=================

* add levenberg-marquardt to available fitting engines

2014-05-01 R0.7.5
=================

* display constraints info on graph
* estimate parameter uncertainty from covariance matrix
* fix windows binary
* read magnetic models from reflpak

2014-04-03 R0.7.4
=================

* demonstrate functional profiles in examples/profile/flayer.py
* add MPI support
* add stopping condition for DE
* support python 2.6, 2.7 and 3.3+
* fix confidence intervals (old confidence intervals are 2x too small)

2013-07-11 R0.7.3
=================

* R0.7.2 broke parallel fitting

2013-06-26 R0.7.2
=================

* support new NCNR reflectometers PBR and Magik
* better labelling of data sets
* monospline fixes
* allow fit interrupt from GUI

2013-05-07 R0.7.1
=================

* simplify contrast variation fits with free variables shared between models
* add FASTA sequence reader with support for labile hydrogen substitution
* redo magnetic profiles so magnetism is a property of nuclear layers
* use material name or layer number to reference model layers
* fix density calculations for natural density
* add support for density and mixtures into chemical formulas

2013-01-25 R0.7.0
=================

* split bumps into its own package
* allow Q probes and oversampling
* allow penalty constraints
* resume a fit from last saved point
* fix garefl and staj file loaders
* fix polarization cross section identifiers
* simulate reflectivity from existing Q,dQ,R,dR data
* show chisq variation in variable histogram

2011-07-28 R0.6.19
==================

First public release
