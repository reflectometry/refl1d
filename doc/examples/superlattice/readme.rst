**********************
Superlattice Models
**********************

.. contents:: :local:

Any structure can be turned into a superlattice using a :class:`refl1d.model.Repeat`.

Simply form a stack as usual, then use that stack within another stack, with a 
repeat modifier.

Here is an example of a multilayer system in the literature:

    Singh, S., Basu, S., Bhatt, P., Poswal, A.K., 
    Phys. Rev. B, 79, 195435 (2009)

In this paper, the authors are interested in the interdiffusion properties 
of Ni into Ti through x-ray and neutron reflectivity measurements. The 
question of alloying at metal-metal interfaces at elevated temperatures is 
critically important for device fabrication and reliability.

.. plot::

    from sitedoc import plot_model
    plot_model('TiNi.py')

The model is defined by :download:`TiNi.py <TiNi.py>`:

.. literalinclude:: TiNi.py
    :linenos:

Within the superlattice definition, the first line defines the repeating 
stack as *bilayer* and the second line uses *bilayer*\ *10 to specify 
10 repeats of the bilayer.

The interface between repeats is defined by the interface at the top 
of the repeating stack, which in this case is the Ti interface.  The 
interface between the superlattice and the next layer is an independent
parameter, whose value defaults to the same initial value as the
interface between the repeats.

If we wanted to have the interface for Ti between repeats identical to 
the interface between Ti and air, we can tie the parameters together::

    sample[1].interface = bilayer[1].interface

If instead we wanted to keep the roughness independent, but start with
a different initial value, we could simply set the interface parameter
value.  In this case, we are setting it to 10 |Ang|\ ::

    sample[1].interface.value = 10

In this model we are fitting all layer and interface widths as well as
the number of repeats.

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

