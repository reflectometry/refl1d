**********************
Superlattice Models
**********************

.. contents:: :local:

Any structure can be turned into a superlattice using a :class:`refl1d.model.Repeat`.

Simply form a stack as usual, then use that stack within another stack, with a
repeat modifier.

.. include:: NiTi.txt

.. include:: PEMU.txt


Freeform structures
===================

The following is a freeform superlattice floating in a solvent
and anchored with a tether molecule.  The tether is anchored via
a thiol group to a multilayer of Si/Cr/Au.  The sulphur in the
thiol attaches well to gold, but not silicon.  Gold will stick
to chrome which sticks to silicon.

Here is the plot using a random tether, membrane and tail group:

.. plot::

    from sitedoc import plot_model
    plot_model('freeform.py')

The model is defined by :download:`freeform.py <freeform.py>`:

.. literalinclude:: freeform.py
    :linenos:
