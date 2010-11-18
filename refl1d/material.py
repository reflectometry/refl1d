# This program is in the public domain
# Author Paul Kienzle
"""
.. sidebar:: On this Page

        * :class:`Base Material <refl1d.material.Material>`
        * :class:`Mixture (mixed block of material) <refl1d.material.Mixture>`
        * :class:`Scattering Length Density (SLD) <refl1d.material.SLD>`
        * :class:`Material Probe Proxy <refl1d.material.ProbeCache>`
        * :class:`Empty Layer (Vaccum) <refl1d.material.Vacuum>`

Reflectometry materials.

Materials (see :class:`Material`) have a composition and a density.
Density may not be known, either because it has not been measured or
because the measurement of the bulk value does not apply to thin films.
The density parameter can be fitted directly, or the bulk density can be
used, and a stretch parameter can be fitted.

Mixtures (see :class:`Mixture`) are a special kind of material which are
composed of individual parts in proportion.  A mixture can be constructed
in a number of ways, such as by measuring proportional masses and mixing
or measuring proportional volumes and mixing.  The parameter of interest
may also be the relative number of atoms of one material versus another.
The fractions of the different mixture components are fitted parameters,
with the remainder of the bulk filled by the final component.

SLDs (see :class:`SLD`) are raw scattering length density values.  These
should be used if the material composition is not known.  In that case,
you will need separate SLD materials for each wavelength and probe.

*air* (see :class:`Vacuum`) is a predefined scatterer transparent to
all probes.

Scatter (see :class:`Scatterer`) is the abstract base class from which
all scatterers are derived.

The probe cache (see :class:`ProbeCache`) stores the scattering factors
for the various materials and calls the material sld method on demand.
Because the same material can be used for multiple measurements, the
scattering factors cannot be stored with material itself, nor does it
make sense to store them with the probe.  The scattering factor lookup
for the material is separate from the scattering length density
calculation so that you only need to look up the material once per fit.

The probe itself deals with all computations relating to the radiation
type and energy.  Unlike the normally tabulated scattering factors f', f''
for X-ray, there is no need to scale by probe by electron radius.  In
the end, sld is just the returned scattering factors times density.
"""
__all__ = ['Material','Mixture','SLD','Vacuum', 'Scatterer', 'ProbeCache']
import sys;
import numpy
from numpy import inf, sqrt, pi
import periodictable
from periodictable.constants import avogadro_number
from mystic import Parameter as Par

class Scatterer(object):
    """
    A generic scatterer separates the lookup of the scattering factors
    from the calculation of the scattering length density.  This allows
    programs to fit density and alloy composition more efficiently.

    .. Note::
       the Scatterer base class is extended by
       :class:`_MaterialStacker <refl1d.model._MaterialStacker>` so that materials
       can be implicitly converted to slabs when used in stack construction
       expressions. It is not done directly to avoid circular dependencies
       between :mod:`model <refl1d.model>` and :mod:`material <refl1d.material>`.
    """
    def sld(self, sf):
        """
        Return the scattering length density expected for the given
        scattering factors, as returned from a call to scattering_factors()
        for a particular probe.
        """
        raise NotImplementedError
    def __str__(self):
        return self.name


# ============================ No scatterer =============================

class Vacuum(Scatterer):
    """
    Empty layer
    """
    name = 'air'
    def parameters(self):
        return []
    def sld(self, probe):
        return 0,0
    def __repr__(self):
        return "Vacuum()"


# ============================ Unknown scatterer ========================

class SLD(Scatterer):
    """
    Unknown composition.

    Use this when you don't know the composition of the sample.  The
    absorption and scattering length density are stored directly rather
    than trying to guess at the composition from details about the sample.

    The complex scattering potential is defined by *rho* + 1j *irho*.
    Note that this differs from *rho* + 1j *mu*/(2 *lambda*) more
    traditionally used in neutron reflectometry, and *N* *re* (*f1* + 1j *f2*)
    traditionally used in X-ray reflectometry.

    Given that *f1* and *f2* are always wavelength dependent for X-ray
    reflectometry, it will not make much sense to uses this for wavelength
    varying X-ray measurements.  Similarly, some isotopes, particularly
    rare earths, show wavelength dependence for neutrons, and so
    time-of-flight measurements should not be fit with a fixed SLD scatterer.
    """
    def __init__(self, name="SLD", rho=0, irho=0):
        self.name = name
        self.rho = Par.default(rho, name=name+" rho" )
        self.irho = Par.default(irho, name=name+" irho" )
    def parameters(self):
        return dict(rho=self.rho, irho=self.irho)
    def sld(self, probe):
        return self.rho.value,self.irho.value

# ============================ Substances =============================

class Material(Scatterer):
    """
    Description of a solid block of material.

    :Parameters:
        *formula* : Formula

            Composition can be initialized from either a string or a chemical
            formula.  Valid values are defined in periodictable.formula.

        *density* : float | |g/cm^3|

            If specified, set the bulk density for the material.

        *natural_density* : float | |g/cm^3|

            If specified, set the natural bulk density for the material.

        *use_incoherent* = False : boolean

            True if incoherent scattering should be interpreted as absorption.

        *fitby* = 'bulk_density' : string

            See :meth:`fitby` for details

        *value* : Parameter or float | units depends on fitby type

            Initial value for the fitted density parameter.  If None, the
            value will be initialized from the material density.

    For example, to fit Pd by cell volume use::

        >>> m = Material('Pd', fitby='cell_volume')
        >>> m.cell_volume.range(10)
        >>> print "Pd density=%.3g volume=%.3g"%(m.density.value,m.cell_volume.value)

    You can change density representation by calling *material.fitby(type)*.

    """
    def __init__(self, formula=None, name=None, use_incoherent=False,
                 density=None, natural_density=None,
                 fitby='bulk_density', value=None):
        self.formula = periodictable.formula(formula, density=density,
                                             natural_density=natural_density)
        self.name = name if name is not None else str(self.formula)
        self.use_incoherent = use_incoherent
        self.fitby(type=fitby, value=value)

    def fitby(self, type='bulk_density', value=None):
        """
        Specify the fitting parameter to use for material density.

        :Parameters:
            *type* : string
                Density representation
            *value* : Parameter
                Initial value, or associated parameter.

        Density type can be one of the following:

            *bulk_density* : |g/cm^3| or kg/L
                Density is *bulk_density*
            *natural_density* : |g/cm^3| or kg/L
                Density is *natural_density* / (natural mass/isotope mass)
            *relative_density* : unitless
                Density is *relative_density* * formula density
            *cell_volume* : |A^3|
                Density is mass / *cell_volume*

        The resulting material will have a *density* attribute with the
        computed material density and the appropriately named attribute.

        .. Note::

            This will delete the underlying parameter, so be sure you specify
            fitby type before using *m.density* in a parameter expression.
            Alternatively, you can use WrappedParameter(m,'density') in your
            expression so that it doesn't matter if fitby is set.
        """

        # Clean out old parameter
        for attr in ('bulk_density','natural_density','cell_volume',
                     'relative_density'):
            try: delattr(self, attr)
            except: pass

        # Put in new parameters
        if type is 'bulk_density':
            if value is None:
                value = self.formula.density
            self.bulk_density = Par.default(value, name=self.name+" density")
            self.density = self.bulk_density
        elif type is 'natural_density':
            if value is None:
                value = self.formula.natural_density
            self.natural_density = Par.default(value, name=self.name+" nat. density")
            self.density = self.natural_density / self.formula.natural_mass_ratio()
        elif type is 'relative_density':
            if value is None:
                value = 1
            self.relative_density = Par.default(value, name=self.name+" rel. density")
            self.density = self.formula.density*self.relative_density
        ## packing factor code should be correct, but radii are unreliable
        #elif type is 'packing_factor':
        #    max_density = self.formula.mass/self.formula.volume(packing_factor=1)
        #    if value is None:
        #        value = self.formula.density/max_density
        #    self.packing_factor = Par.default(value, name=self.name+" packing factor")
        #    self.density = self.packing_factor * max_density
        elif type is 'cell_volume':
            # Density is in grams/cm^3.
            # Mass is in grams/mole.  N_A is 6.02e23 atoms/mole.
            # Volume is in A^3.  1 A is 1e-8 cm.
            if value is None:
                value = self.formula.molecular_mass/self.formula.density
            self.cell_volume = Par.default(value, name=self.name+" cell volume")
            self.density = self.formula.molecular_mass/self.cell_volume
        else:
            raise ValueError("Unknown density calculation type '%s'"%type)

    def parameters(self):
        return {'density': self.density}
    def sld(self, probe):
        rho, irho, incoh = probe.scattering_factors(self.formula)
        if self.use_incoherent:
            raise NotImplementedError("incoherent scattering not supported")
            irho += incoh
        scale = self.density.value
        #print "Material sld ",self.name,scale*coh,scale*absorp
        return (scale*rho,scale*irho)
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Material(%s)"%self.name

class Compound(Scatterer):
    """
    Chemical formula with variable composition.

    :Parameters:

        *parts* : [M1, F1, M2, F2, ...]

    Unlike a simple material which has a chemical formula and a density,
    the formula itself can be varied by fitting the number of atoms of
    each component in the unit cell in addition to the overall density.

    An individual component can be a chemical formula, not just an element.
    """
    def __init__(self, parts=None, density=None, name=None,
                 use_incoherent=False):
        # Split [M1,N1,M2,N2,...] into [M1,M2,...], [N1,N2,...]
        formula = [parts[i] for i in range(0, len(parts),2)]
        count = [parts[i] for i in range(1, len(parts),2)]
        # Convert M1,M2, ... to materials if necessary
        formula = [periodictable.formula(p) for p in formula]
        count = [Par.default(w,limits=(0,inf), name=str(f)+" count")
                  for w,f in zip(count,formula)]
        if name is None: name = "+".join(str(p) for p in formula)
        density = Par.default(density,limits=(0,inf),name=name+" density")

        self.formula = formula
        self.count = count
        self.density = density
        self.name = name
        self.use_incoherent = use_incoherent

        # Save masses so we can recompute number density of compound without
        # having to look up things in the periodic table again.
        self.__mass = numpy.array([f.mass for f in formula])

    def parameters(self):
        """
        Adjustable parameters are the fractions associated with each
        constituent and the relative scale fraction used to tweak
        the overall density.
        """
        return dict(count=self.count, density=self.density)

    def sld(self, probe):
        """
        Return the scattering length density and absorption of the mixture.
        """
        # Convert fractions into an array, with the final fraction
        count = numpy.array([m.value for m in self.count])

        # Lookup SLD assuming density=1, mass=atomic mass
        slds = [probe.scattering_factors(c) for c in self.formula]
        rho,irho,incoh = [numpy.asarray(v) for v in zip(*slds)]

        # coh[i] = N[i]*b_c[i] = density[i]/mass[i] * C[i] * b_c[i]
        # We know density[i]=1 and mass[i] was previously calculated,
        # so we can back out most of the number density calculation,
        # and put it the new number density.
        # The new compound will have SLD of:
        #    density/sum(k*mass) * k*mass[i]*coh[i]
        # Test this by verifying Compound(('H',2,'O',1),density=1)
        # has the sample SLD as Material('H2O',density=1) or some such.
        rho = numpy.sum(rho*(self.__mass*count))
        irho = numpy.sum(irho*(self.__mass*count)[:,None],axis=0)
        if self.use_incoherent:
            raise NotImplementedError("incoherent scattering not supported")
        scale = self.density.value/numpy.sum(count*self.__mass)
        return scale*rho,scale*irho

    def __str__(self):
        return "<%s>"%(",".join(str(M) for M in self.formula))
    def __repr__(self):
        return "Compound([%s])"%(",".join(repr(M) for M in self.formula))


# ============================ Alloys =============================

class _VolumeFraction:
    """
    Returns the relative volume for each component in the system given
    the volume percentages.
    """
    def __init__(self, base, material):
        pass
    def __call__(self, fraction):
        return 0.01*numpy.asarray(fraction)

class _MassFraction:
    """
    Returns the relative volume for each component in the system given
    the relative masses.
    """
    def __init__(self, base, material):
        self._material = [base] + material
    def __call__(self, fraction):
        density = numpy.array([m.density.value for m in self._material])
        volume = fraction/density
        return volume/sum(volume)

class Mixture(Scatterer):
    """
    Mixed block of material.

    The components of the mixture can vary relative to each other, either
    by mass, by volume or by number::

        >>> Mixture.bymass(base,M1,F1,M2,F2...,name='mixture name')
        >>> Mixture.byvolume(base,M1,F1,M2,F2...,name='mixture name')

    The materials *base*, *M1*, *M2*, *M3*, ... can be chemical formula
    strings  or material objects.  In practice, since the chemical
    formula parser does not have a density database, only elemental
    materials can be specified by string. Use natural_density will need
    to change from bulk values if the formula has isotope substitutions.

    The fractions F2, F3, ... are percentages in [0,100]. The implicit
    fraction F1 is 100 - (F2+F3+...). The SLD is NaN when *F1 < 0*).

    name defaults to M1.name+M2.name+...
    """
    @classmethod
    def bymass(cls, base, *parts, **kw):
        """
        Returns an alloy defined by relative mass of the constituents.

        Mixture.bymass(base,M1,F2,...,name='mixture name')
        """
        return cls(base, parts, by='mass', **kw)

    @classmethod
    def byvolume(cls, base, *parts, **kw):
        """
        Returns an alloy defined by relative volume of the constituents.

        Mixture.byvolume(M1,M2,F2,...,name='mixture name')
        """
        return cls(base, parts, by='volume', **kw)

    def __init__(self, base, parts, by='volume', name=None, use_incoherent=False):
        # Split [M1,M2,F2,...] into [M1,M2,...], [F2,...]
        material = [parts[i] for i in range(0, len(parts), 2)]
        fraction = [parts[i] for i in range(1, len(parts), 2)]
        # Convert M1,M2, ... to materials if necessary
        if not isinstance(base,Material): base = Material(base)
        material = [p if isinstance(p,Material) else Material(p)
                    for p in material]

        # Specify the volume calculator based on the type of fraction
        if by == 'volume':
            _volume = _VolumeFraction(base, material)
        elif by == 'mass':
            _volume = _MassFraction(base, material)
        else:
            raise ValueError('fraction must be one of volume, mass or number')

        # Name defaults to names of individual components
        if name is None: name = "+".join(p.name for p in material)

        # Make the fractions into fittable parameters
        fraction = [Par.default(w,limits=(0,100), name=f.name+"% "+by)
                    for w,f in zip(fraction,material[1:])]

        self._volume = _volume
        self.material = material
        self.fraction = fraction
        self.name = name
        self.use_incoherent = use_incoherent

    def parameters(self):
        """
        Adjustable parameters are the fractions associated with each
        constituent and the relative scale fraction used to tweak
        the overall density.
        """
        return dict(base=self.base.parameters(),
                    material=[m.parameters() for m in self.material],
                    fraction=self.fraction,
                    )

    def _density(self):
        """
        Compute the density of the mixture from the density and proportion
        of the individual components.
        """
        fraction = numpy.array([0.]+[m.value for m in self.fraction])
        # TODO: handle invalid fractions using penalty functions
        # S = sum(fraction)
        # scale = S/100 if S > 100 else 1
        # fraction[0] = 100 - S/scale
        # penalty = scale - 1
        fraction[0] = 100 - sum(fraction)
        if (fraction<0).any():
            return NaN
        volume = self._volume(fraction)
        density = numpy.array([m.density() for m in [self.base]+self.material])
        return numpy.sum(volume*density)
    density = property(_density,doc=_density.__doc__)

    def sld(self, probe):
        """
        Return the scattering length density and absorption of the mixture.
        """
        # Convert fractions into an array, with the final fraction
        fraction = numpy.array([f.value for f in self.fraction])
        # TODO: handle invalid fractions using penalty functions
        # S = sum(fraction)
        # scale = S/100 if S > 100 else 1
        # fraction[0] = 100 - S/scale
        # penalty = scale - 1
        if (fraction<0).any():
            return NaN, NaN

        # Lookup SLD
        slds = [c.sld(probe) for c in [self.base] + self.material]
        rho,irho = [numpy.asarray(v) for v in zip(*slds)]

        # Use calculator to convert individual SLDs to overall SLD
        volume_fraction = self._volume(fraction)
        rho = numpy.sum(rho*volume_fraction)
        irho = numpy.sum(irho*volume_fraction)
        if self.use_incoherent:
            raise NotImplementedError("incoherent scattering not supported")
        #print "Mixture",self.name,coh,absorp

        return rho,irho

    def __str__(self):
        return "<%s>"%(",".join(str(M) for M in self.material))
    def __repr__(self):
        return "Mixture(%s)"%(",".join(repr(M) for M in self.material))

# ============================ SLD cache =============================

class ProbeCache:
    """
    Probe proxy for materials properties.

    A caching probe which only looks up scattering factors for materials
    which it hasn't seen before.

    *probe* is the probe to use when looking up the scattering length density.

    The scattering factors need to be retrieved each time the probe
    or the composition changes. This can be done either by deleting
    an individual material from probe (using del probe[material]) or
    by clearing the entire cash.
    """
    def __init__(self, probe=None):
        self._probe = probe
        self._cache = {}
    def clear(self):
        self._cache = {}
    def __delitem__(self, material):
        if material in self._cache:
            del self._cache[material]
    def scattering_factors(self, material):
        """
        Return the scattering factors for the material, retrieving them from
        the cache if they have already been looked up.
        """
        if material not in self._cache:
            self._cache[material] = self._probe.scattering_factors(material)
        return self._cache[material]
