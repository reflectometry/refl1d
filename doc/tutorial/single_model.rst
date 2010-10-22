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
