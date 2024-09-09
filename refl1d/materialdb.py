"""
Common materials in reflectometry experiments along with densities.

By name::

    air, water, heavywater, lightheavywater, silicon, sapphire, gold
    permalloy

By formula::

    H2O, D2O, DHO, Si, Al2O3, Au, Ni8Fe2

If you want to adjust the density you will need to make your own copy of
these materials.  For example, for permalloy::

    >>> NiFe=Material(permalloy.formula, density=permalloy.bulk_density)
    >>> NiFe.density.pmp(10)  # Let density vary by 10% from bulk value
    Parameter(permalloy density)
"""

import periodictable
from .material import Vacuum, Material

__all__ = [
    "air",
    "water",
    "H2O",
    "heavywater",
    "D2O",
    "lightheavywater",
    "DHO",
    "silicon",
    "Si",
    "sapphire",
    "Al2O3",
    "gold",
    "Au",
    "permalloy",
    "Ni8Fe2",
]

# Set the bulk density of heavy water assuming it has the same packing
# fraction as water, but with heavier H[2] substituted for H[1].
rho_D2O = periodictable.formula("D2O").mass / periodictable.formula("H2O").mass
rho_DHO = periodictable.formula("DHO").mass / periodictable.formula("H2O").mass

air = Vacuum()
water = H2O = Material("H2O", density=1, name="water")
heavywater = D2O = Material("D2O", density=rho_D2O)
lightheavywater = DHO = Material("DHO", density=rho_DHO)
silicon = Si = Material("Si")
sapphire = Al2O3 = Material("Al2O3", density=3.965, name="sapphire")
gold = Au = Material("Au", name="gold")
permalloy = Ni8Fe2 = Material("Ni8Fe2", density=8.692, name="permalloy")
