Defining a film
===============

Model scripts are defined using `Python <http://www.python.org>`_.  A
complete introduction to programming and Python is beyond the
scope of this document, and the reader is referred to the many fine
tutorials that exist on the web.

We start with a basic example, a nickel film on silicon:

.. plot::

    import pylab
    from refl1d.fitter import load_problem
    load_problem('examples/ex1/nifilm.py').plot()
    pylab.show()

This model shows three layers (silicon, nickel, and air) as seen in the
solid green line (the step profile).  In addition we have a dashed green
line (the smoothed profile) which corresponds the effective reflectivity
profile, with the $\exp(-2 k_n k_{n+1} \sigma^2)$ interface factored in.

This model is defined by :download:`ex1/nifilm.py </examples/ex1/nifilm.py>`:

.. literalinclude:: /examples/ex1/nifilm.py
    :linenos:

You can preview the model on the command line::

    $ refl1d nifilm.py --preview

Lets break the code down on a line by line basis to understand what is
going on.

1.  Bring in all of the functions from refl1d.names so that we can
    use them in the remainder of the script.

    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 1

3.  Define a new material composed of pure nickel.  The more 
    traditional ``nickel = SLD(rho=9.4)`` could be used instead.

    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 3

4.  Stack the materials (silicon, nickel and air) into a sample.  The
    substrate is silicon with a 5 |Ang| 1-\ $\sigma$ Si:Ni interface.  
    The nickel layer is 100 |Ang| thick with a 5 |Ang| Ni:Air interface.
    Air is on the surface.  Note that silicon and air were predefined in
    refl1d.names.
        
    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 4

6.  Specify which angles we wish to view the reflectivity.  The
    `numpy <http://numpy.scipy.org/>`_ library extends python to
    support vector and matrix operations.  The ``linspace`` function
    above returns values from 0 to 5 in 100 steps for incident angles
    from 0\ |deg| to 5\ |deg|.
        

    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 6

7.  Create a neutron probe.  The probe defines the wavelengths and
    angles which are used for the measurement as well as their
    uncertainties.  From this the resolution of each point can be
    calculated.  We use constants for angular divergence ``dT=0.01``\ |deg|, 
    wavelength ``L=4.75`` |Ang| and wavelength dispersion ``dL=0.0475`` in
    this example, but each angle and wavelength is independent.
        
    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 7

9.  Combine the neutron probe with the sample stack to define an
    experiment.  Using chemical formula and mass density, the same
    sample can be simulated for both neutron and x-ray experiments.

    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 9
	
10. Generate a random data set with 5% noise. While not necessary
    to display a reflectivity curve, it is useful in showing how
    the data set should look.

    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 10

12. Combine a set of experiments into a fitting problem.  The problem
    is used by refl1d for all operations on the model.

    .. literalinclude:: /examples/ex1/nifilm.py
        :lines: 12


Choosing an instrument
======================

Now let's modify the simulation to show how it might look if measured on
the SNS Liquids reflectometer:

.. plot::

    import pylab
    from refl1d.fitter import load_problem
    load_problem('examples/ex1/nifilm-tof.py').plot()
    pylab.show()

This model is defined by 
:download:`ex1/nifilm-tof.py </examples/ex1/nifilm-tof.py>`:

.. literalinclude:: /examples/ex1/nifilm-tof.py
    :linenos:

Here we are using an instrument to control the simulation::

    instrument = SNS.Liquids()
    M = instrument.simulate(sample,
                            T=[0.3,0.7,1.5,3],
                            slits=[0.06, 0.14, 0.3, 0.6],
                            uncertainty = 5,
                            )

The *instrument* line tells us to use the geometry of the SNS Liquids 
reflectometer, which includes information like the distance between the 
sample and the slits and the wavelength range.  We then simulate measurements
of the sample for several different angles *T* (degrees), each with its
own slit opening *slits* (mm).  The simulated measurement duration is
such that the median relative error on the measurement $\Delta R/R$ 
will match *uncertainty* (%).  Because the intensity $I(\lambda)$ varies
so much for a time-of-flight measurement, the central points will be
measured with much better precision, and the end points will be measured
with lower precision.  See 
:meth:`Pulsed.simulate <refl1d.instrument.Pulsed.simulate>` for details 
on all simulation parameters.


Attaching data
==============

We saved the data simulated above into files named :file:`nifilm-tof-1.dat`,
:file:`nifile-tof-2.dat`, etc.  We can load these datasets into a new
model using :download:`ex1/nifilm-data.py </examples/ex1/nifilm-data.py>`:

.. literalinclude:: /examples/ex1/nifilm-data.py
    :linenos:

In this case we are loading multiple data sets into the same
:class:`ProbeSet <refl1d.probe.ProbeSet>` object, which we then associate
with an :class:`Experiment <refl1d.experiment.Experiment>`.  If your
reduction program stitches together the data for you, then you can simply
use ``probe=instrument.load('file')``.

The plot remains the same:

.. plot::

    import pylab
    from refl1d.fitter import load_problem
    load_problem('examples/ex1/nifilm-tof.py').plot()
    pylab.show()

Performing a fit
================

With the sample defined and the data loaded, the last step is to define
the fitting parameters.  This is shown in
:file:`ex1/nifilm-fit.py </examples/ex1/nifilm-fit.py>`:

.. literalinclude:: /examples/ex1/nifilm-fit.py
    :linenos:

Within the sample, layer 0 is the substrate, layer 1 is the film and layer 2
is air.  The interface between two layers is defined by the width of the
interface on top of the layer below.  In this case we are asking for the
best interface values in the range $[0,20]$::

    sample[0].interface.range(0,20)
    sample[1].interface.range(0,20)

We changed the thickness value for this model to start at 125 to give
little challenge to the problem and set the model thickness to this
original value $\pm 50\%$::

    sample[1].thickness.pmp(50)

We can now load and run the fit::

    $ refl1d ex1/nifilm-fit.py --fit=newton --steps=100

.. plot::

    import pylab
    from refl1d.fitter import load_problem, BFGSFit
    p = load_problem('examples/ex1/nifilm-fit.py')
    x = BFGSFit(p).solve(steps=100)
    p.setp(x)
    p.plot()
    pylab.show()
