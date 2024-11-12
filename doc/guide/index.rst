.. _users-guide-index:

############
User's Guide
############

Refl1D is a complex piece of software hiding some simple mathematics.
The reflectivity of a sample is a simple function of its optical
transform matrix $M$.  By slicing the sample in uniform layers, each
of which has a transfer matrix $M_i$, we can estimate the transfer
matrix for a depth-varying sample using $M=\prod M_i$.  We can
adjust the properties of the individual layers until the measured
reflectivity best matches the calculated reflectivty.

The complexity comes from multiple sources:

   * Determining depth structure from reflectivity is an inverse problem
     requiring a search through a landscape with multiple minima, whose
     global minimum is small and often in an unpromising region.
   * The solution is not unique:  multiple minima may be equally valid
     solutions to the inversion problem.
   * The measurement is sensitive to nuisance parameters such as sample
     alignment.  That means the analysis program must include data
     reduction steps, making data handling complicated.
   * The models are complex.  Since the ideal profile is not unique and
     is difficult to locate, we often constrain our search to feasible
     physical models to limit the search space, and to account for
     information from other sources.
   * The reflectivity is dependent on the type of radiation used to probe
     the sample and even its energy.


:ref:`intro-guide`

     Model scripts associate a sample description with data and fitting
     options to define the system you wish to refine.

:ref:`parameter-guide`

     The adjustable values in each component of the system are defined
     by :class:`Parameter <refl1d.mystic.parameter>` objects.  When you
     set the range on a parameter, the system will be able to automatically
     adjust the value in order to find the best match between theory
     and data.

:ref:`data-guide`

     Data is loaded from instrument specific file
     formats into a generic :class:`Probe <refl1d.models.probe.probe.Probe>`.  The
     probe object manages the data view and by extension, the view of
     the theory.  The probe object also knows the measurement resolution,
     and controls the set of theory points that must be evaluated
     in order to computed the expected value at each point.

:ref:`materials-guide`

     The strength of the interaction can be represented either in
     terms of their scattering length density using
     :class:`SLD <refl1d.models.sample.material.SLD>`, or by their chemical
     formula using :class:`Material <refl1d.models.sample.material.Material>`, with
     scattering length density computed from the information in the
     probe.  :class:`Mixture <refl1d.models.sample.material.Mixture>` can be used
     to make a composite material whose parts vary be mass or by volume.

:ref:`sample-guide`

     Materials are composed into samples, usually as a
     :class:`Stack <refl1d.models.sample.layers.Stack>` of
     :class:`Slabs <refl1d.models.sample.layers.Slab>` layers, but more specific profiles
     such as :class:`PolymerBrush <refl1d.models.sample.polymer.PolymerBrush>`
     are available.  Freeform sections of the profile can be described
     using :class:`FreeLayer <refl1d.models.sample.mono.FreeLayer>`, allowing
     arbitrary scattering length density profiles within the layer, or
     :class:`FreeInterface <refl1d.models.sample.mono.FreeInterface>` allowing
     arbitrary transitions from one SLD to another.  New layer types
     can be defined by subclassing :class:`Layer <refl1d.models.sample.layers.Layer>`.

:ref:`experiment-guide`

     Sample descriptions and data sets are combined into an
     :class:`Experiment <refl1d.models.experiment.Experiment>` object,
     allowing the program to compute the expected reflectivity
     from the sample and the probability that reflectivity measured
     could have come from that sample.  For complex cases, where the
     sample varies on a length scale larger than the coherence length
     of the probe, you may need to model your measurement with a
     :class:`CompositeExperiment <refl1d.models.experiment.CompositeExperiment>`.

:ref:`fitting-guide`

     One or more experiments can be combined into a
     :class:`FitProblem <refl1d.models.bumps_interface.FitProblem>`.  This is then
     given to one of the many fitters, such as
     :class:`PTFit <bumps.fitters.PTFit>`, which adjust the varying
     parameters, trying to find the best fit.  PTFit can also
     be used for Bayesian analysis in order to estimate the confidence
     in which the parameter values are known.


.. toctree::
   :maxdepth: 2
   :hidden:

   intro.rst
   parameter.rst
   data.rst
   materials.rst
   sample.rst
   experiment.rst
   fitting.rst
