Simple Models
=============

Model scripts are defined using `Python <http://www.python.org>`_.  A
complete introduction to programming and Python is beyond the
scope of this document, and the reader is referred to the many fine
tutorials that exist on the web.

We start with a basic example, a nickel film on silicon:

.. plot::

    import pylab
    from refl1d.cli import load_job
    load_job(['examples/ex1/nifilm.py']).plot()
    pylab.show()

This model is defined by :download:`nifilm.py </examples/ex1/nifilm.py>`:

.. literalinclude:: /examples/ex1/nifilm.py

You can preview the model on the command line::

    $ refl1d nifilm.py --preview

Lets break the code down on a line by line basis to understand what is
going on.

.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 1

Bring in all of the functions from refl1d.names so that we can
use them in the remainder of the script.

.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 3
	
Define a new material composed of pure nickel.  The more 
traditional ``nickel = SLD(rho=9.4)`` could be used instead.
		
.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 4

Stack the materials (silicon, nickel and air) into a sample.  The
substrate is silicon with a 5 |Ang| 1-\ $\sigma$ Si:Ni interface.  
The nickel layer is 100 |Ang| thick with a 5 |Ang| Ni:Air interface.
Air is on the surface.  Note that silicon and air were predefined in
refl1d.names.

.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 6

Specify which angles we wish to view the reflectivity.  The
`numpy <http://numpy.scipy.org/>`_ library extends python to
support vector and matrix operations.  The ``linspace`` function
above returns values from 0 to 5 in 100 steps for incident angles
from 0\ |deg| to 5\ |deg|.
	
.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 7

We are going to simulate a neutron measurement.  For simplicity,
use an angular divergence ``dT=0.01`` |deg|, a wavelength
``L=4.75`` |Ang| and wavelength dispersion ``dL=0.0475``.  Using
vectors for ``T, dT, L, dL`` the resolution of each point can be
explicitly controlled.
	
.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 9
	
Combine the neutron probe with the sample stack to define an
experiment.  Using chemical formula and mass density, the same
sample can be simulated for both neutron and x-ray experiments.

.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 10
        
Generate a random data set with 5% noise. While not necessary
to display a reflectivity curve, it is useful in showing how
the data set should look.

.. literalinclude:: /examples/ex1/nifilm.py
    :lines: 11
	
Combine a set of experiments into a fitting problem.  The problem
is used by refl1d to for all operations on the model.


Because this
is elemental nickel, we already know it's density.  For compounds
such as 'SiO2' we would have to specify an additional
``density=2.634`` parameter.      


Common materials defined in :mod:`materialdb`::

    *air*, *water*, *silicon*, *sapphire*, ...

Specific elements, molecules or mixtures can be added using the
classes in :mod:`refl1d.material`::

    *SLD*       unknown material with fittable SLD
    *Material*  known chemical formula and fittable density
    *Mixture*   known alloy or mixture with fittable fractions

Materials can be stacked as slabs, with a thickness for each layer and
roughness at the top of each layer.  Because this is such a common
operation, there is special syntax to do it, using '/' to specify
thickness and '%' to specify roughness.  For example, the following
is a 30 A gold layer on top of silicon, with a silicon:gold interface
of 5 A and a gold:air interface of 2 A:

    >> from refl1d import *
    >> sample = silicon(0,5) | gold(30,2) | air
    >> print sample
    Si | Au(30) | air

Individual layers and stacks can be used in multiple models, with all
parameters shared except those that are explicitly made separate.  The
syntax for doing so is similar to that for lists.  For example, the
following defines two samples, one with Si+Au/30+air and the other with
Si+Au/30+alkanethiol/10+air, with the silicon/gold layers shared::


    >> alkane_thiol = Material('C2H4OHS',bulk_density=0.8,name='thiol')
    >> sample1 = silicon(0,5) | gold(30,2) | air
    >> sample2 = sample1[:-1] | alkane_thiol(10,3) | air
    >> print sample2
    Si | Au(30) | thiol(10) | air

Stacks can be repeated using a simple multiply operation.  For example,
the following gives a cobalt/copper multilayer on silicon::

    >> Cu = Material('Cu')
    >> Co = Material('Co')
    >> sample = Si | [Co(30) | Cu(10)]*20 | Co(30) | air
    >> print sample
    Si | [Co(30) | Cu(10)]*20 | Co(30) | air

Multiple repeat sections can be included, and repeats can contain repeats.
Even freeform layers can be repeated.  By default the interface between
the repeats is the same as the interface between the repeats and the cap.
The cap interface can be set explicitly.  See :class:`model.Repeat` for
details.


.. _single-model:


############################
Single Model and Single Data
############################


Setting up Models
=================
Reflectometry models consist of 1-D stacks of layers. 
Layers are joined by gaussian interfaces. The layers
themselves may be uniform, or the scattering density 
may vary with depth in the layer.

A model consists of the following:

* A semi-infinite flat layer on the left.
* A semi-infinite flat layer on the right.
* Any number of layers in between.
* A roughness value at each interface between layers.

Roughness
=============

The roughness is used to smooth the interfaces between layers.
Each interface between layers has an associated roughness. Hence,
if there are N user-specified layers (in addition to the two 
semi-infinite layers), there are N+1 roughness. Note that each 
layer has both a "Left Rough" (the roughness on its left interface) 
and a "Right Rough" (the roughness on its right interface). The
smoothing uses an error function where $\sigma$ is equal to the
roughness. Hence, two adjacent flat layers with a roughness of zero
(indicating no smoothing) will look like a step function,
while if there is a positive roughness it will look like
an error function. It is important to note that the smoothing 
only passes through one layer. hence, if the depth of a 
layer is very small relative to the roughness at its 
interfaces, discontinous behavior can result.


Displaying Models
=================

Adding data
=================

Displaying data
=================

Setting Fit Parameters
======================

Running Fit
=================
