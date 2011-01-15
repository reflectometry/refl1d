First define materials that you are going to use.

Common materials defined in :module:`materialdb`::

    *air*, *water*, *silicon*, *sapphire*, ...

Specific elements, molecules or mixtures can be added using the
classes in :module:`material`::

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
++++++++++

The roughness is used to smooth the interfaces between layers.
Each interface between layers has an associated roughness. Hence,
if there are N user-specified layers (in addition to the two 
semi-infinite layers), there are N+1 roughness. Note that each 
layer has both a "Left Rough" (the roughness on its left interface) 
and a "Right Rough" (the roughness on its right interface). The
smoothing uses an error function where sigma is equal to the
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
