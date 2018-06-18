**************
Change History
**************

Note: updated docs now found at `<http://refl1d.readthedocs.org>`_

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
