# This program is public domain
# Author: Paul Kienzle
"""
Composition space modeling
"""

from mystic.parameter import Parameter as Par

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
        if name == None: name="solvent "+solvent.name
        self.name = name
        self.thickness = Par.default(thickness, limits=(0,inf),
                                   name=name+" thickness")
    def parameters(self):
        return dict(solvent=self.solvent.parameters(),
                    thickness=self.thickness,
                    parts=[p.parameters() for p in self.parts])

    def add(self, part=None):
        self.parts.append(part)
    # Array style interface to the parts
    def __getitem__(self, n):
        return self.parts[n]
    def __delitem__(self, n):
        del self.parts[n]
    def __setitem__(self, n, part):
        self.parts[n] = part
    def plot_volume_fraction(self, ax):
        """
        Composition space items have a plotting routine for showing the
        volume profile.
        """
        # Uniform stepping
        z = arange(slabs.dz/2, self.thickness.value, slabs.dz)

        # Storage for the sub-totals
        volume_total = numpy.zeros_like(z)

        # Accumulate the parts
        for p in self.parts:
            f = p.profile(z)
            ax.plot(z,f,label=p.name)
            volume_total += f

        # Remainder is solvent
        ax.plot(z,1-volume_total,label=solvent.name)

    # Render a profile
    def render(self, probe, slabs):
        # Uniform stepping
        z = arange(slabs.dz/2, self.thickness.value, slabs.dz)

        # Storage for the sub-totals
        n,k = len(z), slabs.nprobe
        rho_total = numpy.zeros((n,k))
        mu_total = numpy.zeros((n,k))
        volume_total = numpy.zeros_like(z)

        # Accumulate the parts
        for p in self.parts:
            f, rho, mu = p.f_sld(probe,z)
            rho_total += rho
            mu_total += mu
            volume_total += f

        # Remainder is solvent
        rho,mu = probe.sld(self.solvent)
        rho_total += rho*(1-volume_total)
        mu_total += mu*(1-volume_total)

        # Add to model
        w = slabs.dz * numpy.ones(size(z))
        slabs.extend(w=w,rho=rho_total,mu=mu_total)

class Part:
    def __init__(self, material, profile, fraction=1):
        self.material = material
        self.profile = profile
        self.fraction = Par.default(fraction, limits=(0,1),
                                  name=self.material.name+" fraction")
    def parameters(self):
        return dict(material=self.material.parameters(),
                    profile=self.profile.parameters(),
                    fraction=self.fraction)
    def f_sld(self, probe, z):
        # Note: combining f and sld because there my be some
        # composites such as oriented proteins for which the
        # sld and volume change at the same time.
        rho,mu = probe.sld(self.material)
        f = fraction.value*self.profile(z)
        return f, rho*f, mu*f

class TetheredPolymer(Profile):
    """
    Tethered polymer profile::

        1                         if z in (-inf, h)
        (1 - ((z-h)/Lo)**2)**y    if z in (h, h+Lo)
        0                         if z in (h+Lo, inf)

    where h is the length of the head group and Lo is the length of the
    tail group.

    Parameters::

        *head* is h in [0,inf)
        *tail* is Lo in [0,inf)
        *power* is y in [0,inf)

    This profile will be used to construct a volume fraction component
    which a scaling fraction phi and a material describing the scattering
    properties of the profile.

    Kent, et al. (1999) Tethered chains in poor solvent conditions: An
    experimental study involving Langmuir diblock copolymer monolayers
    J Chem Phys 110(7) 3553-3565.
    """
    def __init__(self, head=0, tail=0, power=0.5, name="brush"):
        self.head = Par.default(head, limits=(0,inf),
                              name=name+" head")
        self.tail = Par.default(tail, limits=(0,inf),
                              name=name+" tail")
        self.power = Par.default(power, limits=(0,inf),
                               name=name+" power")
    def parameters(self):
        return dict(head=self.head, tail=self.tail, power=self.power)
    def __call__(self, z):
        head,tail,power = self.head.value, self.tail.value, self.power.value
        if power <= 0:
            return 1. * (z < head+tail)
        else:
            return (1 - clip((z-head)/tail,0,1)**2)**power

class Gaussian(Profile):
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
        self.center = Par.default(center, limits=(-inf,inf),
                                name=name + " center")
        self.width = Par.default(width, limits=(0,inf),
                               name=name + " width")
        self.stretch = Par.default(sigma, limits=(0,inf),
                                 name=name + " sigma")
    def parameters(self):
        return dict(center=self.center, width=self.width, sigma=self.sigma)
    def __call__(self, z):
        mu,sigma,w = self.center.value, self.sigma.value, self.width.value/2
        if w <= 0:
            result = exp(-0.5*((z-mu)/sigma)**2)
        else:
            result = numpy.ones_like(z)
            idx = abs(z-mu) < w
            result[idx] = exp(-0.5*((abs(z[idx]-mu)-w)/sigma)**2)
        return result
