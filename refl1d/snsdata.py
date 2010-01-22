# This program is in the public domain
# Author: Paul Kienzle
"""
SNS data loaders.

The following instruments are defined::

    Liquids, Magnetic

These are :class:`resolution.Polychromatic` classes tuned with 
default instrument parameters and loaders for reduced SNS data.
See :module:`resolution` for details.
"""

from .resolution import Polychromatic

print "Insert correct slit defaults for Liquids and Magnetic"
class Liquids(Polychromatic):
    """
    Loader for reduced data from the SNS Liquids instrument.
    """
    instrument = "Liquids"
    radiation = "neutron"
    wavelength = 2.5,17.5
    dLoL = 0.02
    d_s1 = 230.0 + 1856.0
    d_s2 = 230.0

class Magnetic(Polychromatic):
    """
    Loader for reduced data from the SNS Magnetic instrument.
    """
    instrument = "Magnetic"
    radiation = "neutron"
    wavelength = 1.8,14
    dLoL = 0.02
    d_s1 = 75*2.54
    d_s2 = 14*2.54
