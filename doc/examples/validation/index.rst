.. _index:

Polarized Reflectivity: Gepore Validation
==========================================
We will use the `gepore.f` program (Fortran77), as published in
the PNR book chapter by Majkrzak et al. [1]_
to validate the results of the Refl1D magnetic calculation kernel.

For models expected Zeeman corrections 
(with large applied field `H` and non-zero in-plane component of `M` perpendicular to `H`), 
we will use the version `gepore_zeeman.f` which was created during the preparation of an
article on PNR in large magnetic fields in the Journal of Applied Crystallography [2]_

.. toctree::
    :maxdepth: 1

    ../../notebooks/gepore_sf_outofplane.ipynb
    ../../notebooks/gepore_sf_inplane.ipynb
    ../../notebooks/gepore_nsf.ipynb

.. [1] Majkrzak, C. F., K. V. O'Donovan, and N. F. Berk.
    "Polarized neutron reflectometry."
    Neutron Scattering from Magnetic Materials. Elsevier Science, 2006. 397-471.
    https://doi.org/10.1016/B978-044451050-1/50010-0

.. [2] Maranville, B. B., Kirby, B. J., Grutter, A. J., Kienzle, P. A., Majkrzak, C. F., Liu, Y., and Dennis, C. L.
    "Measurement and modeling of polarized specular neutron reflectivity in large magnetic fields."
    Journal of Applied Crystallography 49.4 (2016): 1121-1129.
    https://doi.org/10.1107/S1600576716007135
