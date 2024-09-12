# This program is public domain
# Author: Paul Kienzle
"""
Composition space modeling

INCOMPLETE UNUSED UNTESTED CODE


"""

#@PydevCodeAnalysisIgnore

import numpy as np
from numpy import inf, exp
from bumps.parameter import Parameter as Par, to_dict

from .model import Layer
from .materialdb import air

class CompositionSpace(Layer):
    """
    A composition space is much like a stack except that the
    components of the composition space can slide past each
    other, and the parts not taken up by the components is filled
    with a solvent.
    """
    def __init__(self, solvent=air, thickness=0, name=None):
        self.parts = []
        self.solvent = solvent
        if name is None:
            name = "solvent "+solvent.name
        self.name = name
        self.thickness = Par.default(thickness, limits=(0, inf),
                                     name=name+" thickness")

    def parameters(self):
        return {
            'solvent': self.solvent.parameters(),
            'parts': [p.parameters() for p in self.parts],
        }

    def to_dict(self):
        return to_dict({
            'type': type(self).__name__,
            'name': self.name,
            'solvent': self.solvent,
            'parts': self.parts,
        })

    def add(self, part=None):
        self.parts.append(part)

    # Array style interface to the parts
    def __getitem__(self, n):
        return self.parts[n]

    def __delitem__(self, n):
        del self.parts[n]

    def __setitem__(self, n, part):
        self.parts[n] = part

    def plot_volume_fraction(self, slabs, ax):
        """
        Composition space items have a plotting routine for showing the
        volume profile.
        """
        # Uniform stepping
        z = np.arange(slabs.dz/2, self.thickness.value, slabs.dz)

        # Storage for the sub-totals
        volume_total = np.zeros_like(z)

        # Accumulate the parts
        for p in self.parts:
            f = p.profile(z)
            ax.plot(z, f, label=p.name)
            volume_total += f

        # Remainder is solvent
        ax.plot(z, 1-volume_total, label=self.solvent.name)

    # Render a profile
    def render(self, probe, slabs):
        # Uniform stepping
        z = np.arange(slabs.dz/2, self.thickness.value, slabs.dz)

        # Storage for the sub-totals
        n, k = len(z), slabs.nprobe
        rho_total = np.zeros((n, k))
        mu_total = np.zeros((n, k))
        volume_total = np.zeros_like(z)

        # Accumulate the parts
        for p in self.parts:
            f, rho, mu = p.f_sld(probe, z)
            rho_total += rho
            mu_total += mu
            volume_total += f

        # Remainder is solvent
        rho, mu = probe.sld(self.solvent)
        rho_total += rho*(1-volume_total)
        mu_total += mu*(1-volume_total)

        # Add to model
        w = slabs.dz * np.ones(z.shape)
        slabs.extend(w=w, rho=rho_total, mu=mu_total)

class Part(object):
    def __init__(self, material, profile, fraction=1):
        self.material = material
        self.profile = profile
        self.fraction = Par.default(fraction, limits=(0, 1),
                                    name=self.material.name+" fraction")

    def parameters(self):
        return {
            'material': self.material.parameters(),
            'profile': self.profile.parameters(),
            'fraction': self.fraction,
        }

    def to_dict(self):
        return to_dict({
            'type': type(self).__name__,
            'material': self.material,
            'profile': self.profile,
            'fraction': self.fraction,
        })

    def f_sld(self, probe, z):
        # Note: combining f and sld because there my be some
        # composites such as oriented proteins for which the
        # sld and volume change at the same time.
        rho, mu = probe.sld(self.material)
        f = self.fraction.value*self.profile(z)
        return f, rho*f, mu*f


class Gaussian(object):
    """
    Stretched gaussian profile::

        exp ( -(|z-mu|-w)**2 / 2 sigma**2 )   if |z-mu| > w
        1                                     otherwise

    Parameters::

        *center* is mu in (-inf, inf)
        *sigma* is sigma in [0, inf)
        *width* is 2*w in [0, inf)

    Use this profile to represent a flat profile component which is
    subject to gaussian noise in its position in the composition space.

    This profile will be used to construct a volume fraction component
    which a scaling fraction phi and a material describing the scattering
    properties of the profile.
    """
    def __init__(self, center=0, width=0, sigma=1, name="gauss"):
        self.center = Par.default(center, limits=(-inf, inf),
                                  name=name + " center")
        self.width = Par.default(width, limits=(0, inf),
                                 name=name + " width")
        self.stretch = Par.default(sigma, limits=(0, inf),
                                   name=name + " sigma")

    def parameters(self):
        return {'center':self.center, 'width':self.width, 'sigma':self.stretch}

    def to_dict(self):
        ret = { 'type': type(self).__name__, }
        ret.update(to_dict(self.parameters()))
        return ret

    def __call__(self, z):
        mu, sigma, w = self.center.value, self.stretch.value, self.width.value/2
        if w <= 0:
            result = exp(-0.5*((z-mu)/sigma)**2)
        else:
            result = np.ones_like(z)
            idx = abs(z-mu) < w
            result[idx] = exp(-0.5*((abs(z[idx]-mu)-w)/sigma)**2)
        return result
