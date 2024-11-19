.. _intro-guide:

*******************
Using Refl1D
*******************

.. contents:: :local:

The Refl1D library is organized into modules.  Specific functions and
classes can be imported from a module, such as::

    >>> from refl1d.sample.layers import Slab

The most common imports have been gathered together in refl1d.  This
allows you to use names like :class:`Slab <refl1d.sample.layers.Slab>` directly::

    >>> from refl1d.names import *
    >>> s = Slab(silicon, thickness=100, interface=10)

This pattern of importing all names from a file,  while convenient for
simple scripts, makes the code more difficult to understand later, and
can lead to unexpected results when the same name is used in multiple
modules.  A safer, though more verbose pattern is to use:

    >>> import refl1d as ref
    >>> s = ref.sample.layers.Slab(ref.sample.materialdb.silicon, thickness=100, interface=10)

This documents to the reader unfamiliar with your code (such as you when
looking at your model files two years from now) exactly where the
name comes from.

