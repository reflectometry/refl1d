**********************
Superlattice Models
**********************

.. contents:: :local:

Any structure can be turned into a superlattice using a :class:`refl1d.model.Repeat`.

Simply form a stack as usual, then use that stack within another stack, with a
repeat modifier.

Hard material structures
=========================

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


Soft material structures
=========================

Another example involves the inter-diffusion of polyelectrolyte polymer multi-layers.

Reference: Jomaa, H., Schlenoff, Macromolecules, 38 (2005), 8473-8480

Inter-diffusion properties of multilayer systems are of great interest in both hard and soft materials. Jomaa, et. al  has showed that reflectometry can be used to elucidate the kinetics of a diffusion process in polyelectrolytes multi-layers. Although the purpose of this paper was not to fit the presented system, it offers a good model for an experimentally relevant system for which information from neutron reflectometry can be obtained. In this model system we will show that we can create a model for this type of system and determine the relevant parameters through our optimisation scheme. This particular example uses deuterated reference layers to determine the kinetics of the overall system. 


To model the system described in figure 2 of the reference, we do the following:

    .. literalinclude:: PEMU.py
        :lines: 8,13-14,17,20,23,26,29,32,33,36-38,41-42,45-46

The steps are detailed as follows:

1.  Bring in all of the functions from refl1d.names so that we can use them in the remainder of the script.

    .. literalinclude:: PEMU.py
        :lines: 8

2.	The polymer system is deposited on a gold film with chromium as an adhesion layer. Because these are standard films which are very well-known in this experiment we can use the build-in materials library to create these layers.

	.. literalinclude:: PEMU.py
		:line: 13-14

3.	Now we can created the PDADMA/dPSS layer. In this particular example, the two materials are modelled as a single layer and the make-up of this material is complicated so it is more convenient to use the scattering length density(SLD) functionality. We assume calculations were carried out to estimate the SLD value. We also have an idea of the layer thickness based on ellipsometery(as stated in the paper).

	.. literalinclude:: PEMU.py
		:line: 17

4.	We now do the same for the non-deuterated PDADMA/PAA layer. Again, the two layers are estimated to have only a single SLD and so only one layer is created. Similar knowledge about the SLD and thickness is assumed:
	
	.. literalinclude:: PEMU.py
		:line: 20

5.	Now we can create the repeat stack. This is accomplished quite conveniently by using the | notation to create a variable 'bilayer':

	.. literalinclude:: PEMU.py
		:line: 23

6. 	Now we can build the model sample. We can use the bilayer variable and repeat it as seen below:

 	.. literalinclude:: PEMU.py
		:line: 26
	
.. Note:: In this system we expect the kinetics of the surface diffusion to differ from that of the bulk layer structure. Because we want the top bilayer to optimise independently of the other bilayers, the fifth layer was not included in the stack. If the diffusion properties of each layer were expected to vary widely from one-another, the stack would have to forgo the refl1d Repeat functionality.
 
7. Now that the model sample is built, we can start adding ranges to the fit parameters. We assume that the chromium/gold layer is well known through other methods and will not fit it; however, additional optimisation could certainly be included here. We assume all interfaces in the bilayer act in the same way and that they could potentially diffuse a significant distance.

	.. literalinclude:: PEMU.py
		:line: 29

8. 	We also expect, as diffusion occurs, the SLD of the deuterated polymer will decrease and the undeuterated segment will increase. These are seen in the limits chosen for the optimiser:

	.. literalinclude:: PEMU.py
		:line: 32-33

9. 	Finally, limits need to be assigned to the top bilayer. Although we want the optimiser to threat these parameters independently because surface diffusion is expected to occur faster, the overall nature of the diffusion is expected to be the same and so we use the same limits.

	.. literalinclude:: PEMU.py
		:line: 36-38

10.	Now we can finish preparing the experiment. The probe is a neutron probe and so we have:

	.. literalinclude:: PEMU.py
		:line: 41-42

11. We can now create the experiment, simulate the data, and finish the fit problem:
	
	.. literalinclude:: PEMU.py
		:line: 45-48

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
