.. _model:

######################
Layer models
######################

Reflectometry samples consist of 1-D stacks of layers joined by error
function interfaces. The layers themselves may be uniform slabs, or 
the scattering density may vary with depth in the layer.  The first
layer in the stack is the substrate and the final layer is the surface.
Surface and substrate are assumed to be semi-infinite, with any thickness
ignored.

Interfaces
=============

The interface between layers is assumed to smoothly follow and
error function profile to blend the layer above with the layer below.
The interface value is the 1-\ $\sigma$ gaussian roughness.
Adjacent flat layers with zero interface will act like a step function,
while positive values will introduce blending between the layers.

Blending is usually done with the Nevot-Croce formalism, which scales
the index of refraction between two layers by $\exp(-2 k_n k_{n+1} \sigma^2)$.
We show both a step function profile for the interface, as well as the 
blended interface.  

.. note:: 

    The blended interface representation is limited to the neighbouring 
    layers, and is not an accurate representation of the effective 
    reflectivity profile when the interface value is large relative to 
    the thickness of the layer.  

We will have a mechanism to force the use of the blended profile for
direct calculation of the interfaces rather than using the interface
scale factor.


Slabs
============

Materials can be stacked as slabs, with a thickness for each layer and
roughness at the top of each layer.  Because this is such a common
operation, there is special syntax to do it, using '|' as the layer
separator and `()` to specify thickness and interface.  For example, 
the following is a 30 |Ang| gold layer on top of silicon, with a 
silicon:gold interface of 5 |Ang| and a gold:air interface of 2 |Ang|::

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


