.. _polarized_xs_order:

Polarized Cross Sections Order Error
=====================================

In versions of Refl1D before 1.0.0, there was an error in the interpretation of the outputs of the
magnetic calculation kernel.
We were inadvertently using an opposite convention for the direction of the `Aguide`
angle than what is used for the `EPS` angle in the Polarized Neutron Reflectometry (PNR)
book chapter by Majkrzak et al. [1]_, while simultaneously using
the outputs of the magnetic calculation kernel in reversed order.

Choosing an opposite convention for the `Aguide` angle causes the outputs of the calculation
to be reversed when there is no splitting in the spin-flip cross sections, so these two
choices effectively cancel each other out almost all the time.

The exception is when there is significant splitting in the spin-flip cross sections,
which occurs when there is a significant applied field **and** a significant projection of the
magnetic moment perpendiculat to that applied field.  In this case, the energy corrections
are applied to the wrong elements, leading to incorrect results.

We have corrected this error in the release 1.0.0 of Refl1D.

*Note that as a result of the original error in interpreting the outputs of the Refl1D
magnetic kernel, the reflectivity figures in the original Journal of Applied Crystallography
article on PNR* [2]_ *in large fields have the cross sections labeled in reverse order.*

Backward Compatibility
----------------------

To maintain backward compatibility, the `Aguide` parameter in the model is now interpreted as
:math:`\text{Aguide} \equiv -\text{EPS}`, where :math:`\text{EPS}` is the angle between the 
sample normal (z-axis) and the applied guide field, as shown in figure 1.114 in the book chapter.

The output of the magnetic kernel is in the order :math:`R^{++}, R^{+-}, R^{-+}, R^{--}`, 
but the output of `refl1d.sample.reflectivity.magnetic_amplitude` is reversed to match the order
of the cross sections as stored in `refl1d.probe.PolarizedNeutronProbe`: 
:math:`R^{--}, R^{-+}, R^{+-}, R^{++}`.

For existing models, unless there is splitting in the spin-flip cross sections, the calculated
results will remain the same as before.

Validation
----------

See :doc:`Gepore Validation </notebooks/gepore_sf>` example for a comparison of the results including the
correction.

References
----------

.. [1] Majkrzak, C. F., K. V. O'Donovan, and N. F. Berk.
    "Polarized neutron reflectometry."
    Neutron Scattering from Magnetic Materials. Elsevier Science, 2006. 397-471.
    https://doi.org/10.1016/B978-044451050-1/50010-0

.. [2] Maranville, B. B., Kirby, B. J., Grutter, A. J., Kienzle, P. A., Majkrzak, C. F., Liu, Y., and Dennis, C. L.
    "Measurement and modeling of polarized specular neutron reflectivity in large magnetic fields."
    Journal of Applied Crystallography 49.4 (2016): 1121-1129.
    https://doi.org/10.1107/S1600576716007135
