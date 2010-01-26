# This program is in the public domain
# Author Paul Kienzle
"""
Reflectometry materials.

Materials (see :class:`Material`) have a composition and a density.
Density may not be known, either because it has not been measured or
because the measurement of the bulk value does not apply to thin films.
The density parameter can be fitted directly, or the bulk density can be
used, and a stretch parameter can be fitted.

Mixtures (see :class:`Mixture`) are a special kind of material which are
composed of individual parts in proportion.  A mixture can be construct
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
__all__ = ['Material','Mixture','SLD','Vacuum']

import numpy
from numpy import inf
import periodictable
from periodictable.constants import avogadro_number
from mystic import Parameter as Par

_INCOHERENT_AS_ABSORPTION = True

class Scatterer:
    """
    A generic scatterer separates the lookup of the scattering factors
    from the calculation of the scattering length density.  This allows
    programs to fit density and alloy composition more efficiently.

    Note: the scatterer base class is extended in model so that materials
    can be implicitly converted to slabs when used in stack construction
    expressions.
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

    Clearly, all bets are off regarding probe dependent effects, and
    each probe and wavelength will need a separate material.
    """
    def __init__(self, name="X", rho=0, mu=0):
        self.name = name
        self.rho = Par.default(rho, name=name+" rho" )
        self.mu = Par.default(mu, name=name+" mu" )
    def parameters(self):
        return dict(rho=self.rho, mu=self.mu)
    def sld(self, probe):
        return self.rho.value,self.mu.value

# ============================ Substances =============================

class Material(Scatterer):
    """
    Description of a solid block of material.

    *formula* (chemical formula)

        Composition can be initialized from either a string or a chemical
        formula.  Valid values are defined in periodictable.formula.

    *density* (None g/cm**3)

        If specified, set the bulk density for the material

    *packing_factor* (None)

        If specified, set the bulk density for the material from the
        packing factor or from the named lattice type (bcc, fcc, hcp,
        cubic, or diamond).

    *use_incoherent* (False)

        True if incoherent scattering should be treated as absorption.

    *fitby* (bulk_density)

        Indicate how density should be represented in the fit.  This should
        be one of bulk_density (g/cm**3), relative_density (unitless),
        cell_volume (A**3) or packing_factor (unitless).

    For example, to fit Pd by cell volume use::

        m = Material('Pd', fitby='cell_volume')
        m.cell_volume.range(10)
        print "Pd density",m.density.value

    You can change density representation by calling fitby(type).  Note that
    this will delete the underlying parameter, so be sure you specify fitby
    type before using m.density in a parameter expression.  Alternatively,
    you can use WrappedParameter(m,'density') in your expression so that it
    doesn't matter if fitby is set.

    **WARNING** as of this writing packing_factor does not seem to return
    the correct density.
    """
    def __init__(self, formula=None, name=None,
                 incoherent_as_absorption=_INCOHERENT_AS_ABSORPTION,
                 density=None, packing_factor=None, fitby='bulk_density'):
        self.formula = periodictable.formula(formula, density=density)
        self.name = name if name is not None else str(self.formula)
        self.incoherent_as_absorption = incoherent_as_absorption
        self.fitby(fitby)

    def fitby(self, type):
        """
        Specify the fitting parameter to use for material density.

        Only one of the following density parameters can be fit:

            bulk_density (g/cm**3 or equivalently, kg/L)
            cell_volume (A**3)
            packing_factor (unitless)
            relative_density (unitless: density relative to bulk density)

        The default is to fit the bulk density directly.

        """
        #TODO: test cell_volume and packing_factor

        # Clean out old parameter
        for attr in ('bulk_density','cell_volume',
                     'relative_density','packing_factor'):
            try: delattr(self, attr)
            except: pass

        # Put in new parameters
        if type is 'bulk_density':
            self.bulk_density = Par.default(self.formula.density,
                                            name=self.name+" density")
            self.density = self.bulk_density
        elif type is 'relative_density':
            self.relative_density = Par.default(1, name=self.name+" stretch")
            self.density = self.formula.density*self.relative_density
        elif type is 'packing_factor':
            max_density = self.formula.mass/self.formula.volume(packing_factor=1)
            pf = self.formula.density/max_density
            self.packing_factor = Par.default(pf, name=self.name+" packing factor")
            self.density = self.packing_factor * max_density
        elif type is 'cell_volume':
            # Density is in grams/cm**3.
            # Mass is in grams/mole.  N_A is 6.02e23 atoms/mole.
            # Volume is in A**3.  1 A is 1e-8 cm.
            units = 1e24/avogadro_number
            vol = (self.formula.mass*units)/self.formula.density
            self.cell_volume = Par.default(vol, name=self.name+" cell volume")
            self.density = (self.formula.mass*units)/self.cell_volume
        else:
            raise ValueError("Unknown density calculation type '%s'"%type)

    def parameters(self):
        return {'density': self.density}
    def sld(self, probe):
        coh, absorp, incoh = probe.scattering_factors(self.formula)
        if self.incoherent_as_absorption:
            absorp = absorp + incoh
        scale = self.density.value
        #print "Material sld ",self.name,scale*coh,scale*absorp
        return (scale*coh,scale*absorp)
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Material(%s)"%self.name


class Compound(Scatterer):
    """
    Chemical formula with variable composition.

    Unlike a simple material which has a chemical formula and a density,
    the formula itself can be varied in a compound.  In particular, the
    absolute number of atoms of each component in the unit cell can be a
    fitted parameter, in addition to the overall density.

    An individual component can be a chemical formula, not just an element.
    """
    def __init__(self, parts=None, density=None, name=None,
                 incoherent_as_absorption=_INCOHERENT_AS_ABSORPTION):
        # Split [M1,N1,M2,N2,...] into [M1,M2,...], [N1,N2,...]
        formula = [parts[i] for i in range(0, len(parts),2)]
        count = [parts[i] for i in range(1, len(parts),2)]
        # Convert M1,M2, ... to materials if necessary
        formula = [periodictable.formula(p) for p in formula]
        count = [Par.default(w,limits=(0,inf), name=str(f)+" count")
                  for w,f in zip(count,formula)]
        if name is None: name = "+".join(str(p) for p in formula)
        density = Par.default(density,limits=(0,inf),name=name+" density")

        self.incoherent_as_absorption = incoherent_as_absorption
        self.formula = formula
        self.count = count
        self.density = density
        self.name = name

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
        coh,absorp,incoh = [numpy.asarray(v) for v in zip(*slds)]
        if self.incoherent_as_absorption:
            absorp = absorp + incoh[:,None]

        # coh[i] = N[i]*b_c[i] = density[i]/mass[i] * C[i] * b_c[i]
        # We know density[i]=1 and mass[i] was previously calculated,
        # so we can back out most of the number density calculation,
        # and put it the new number density.
        # The new compound will have SLD of:
        #    density/sum(k*mass) * k*mass[i]*coh[i]
        # Test this by verifying Compound(('H',2,'O',1),density=1)
        # has the sample SLD as Material('H2O',density=1) or some such.
        coh = numpy.sum(coh*(self.__mass*count))
        absorp = numpy.sum(absorp*(self.__mass*count)[:,None],axis=0)
        scale = self.density.value/numpy.sum(count*self.__mass)
        #print "Compound sld",self.name,scale*coh,scale*absorp
        return (scale*coh,scale*absorp)

    def __str__(self):
        return "<%s>"%(",".join(str(M) for M in self.formula))
    def __repr__(self):
        return "Compound([%s])"%(",".join(repr(M) for M in self.formula))


# ============================ Alloys =============================

class _VolumeFraction:
    """
    Returns the relative volume for each component in the system given
    the relative volumes.  Clearly, this is just the identity function.
    """
    def __init__(self, material):
        pass
    def __call__(self, fraction):
        return 0.01*numpy.asarray(fraction)

class _MassFraction:
    """
    Returns the relative volume for each component in the system given
    the relative masses.
    """
    def __init__(self, material):
        self.material = material
    def __call__(self, fraction):
        density = numpy.array([m.density.value for m in self.material])
        volume = fraction/density
        return volume/sum(volume)

class Mixture(Scatterer):
    """
    Mixed block of material.

    The components of the mixture can vary relative to each other, either
    by mass, by volume or by number::

        Mixture.bymass(M1,F1,M2,F2,...,name='mixture name')
        Mixture.byvolume(M1,F1,M2,F2,...,name='mixture name')

    The materials M1, M2, ... can be chemical formula strings or material
    objects.  In practice, since the chemical formula parser does not have
    a density database, only elemental materials can be specified by
    densities will be valid.  Be aware that density will need to change from
    bulk values if the formula has isotope substitutions.

    The fractions F1, F2, ... are positive real numbers.  These will be
    converted to percentages when the mixture is defined, and so will
    have a fit range of [0,100].  The first fraction is not fittable, and
    will adjusted so that the total volume of the material sums to 100%.

    name defaults to M1.name+M2.name+...
    """
    @classmethod
    def bymass(cls, *parts, **kw):
        """
        Returns an alloy defined by relative mass of the constituents.

        Mixture.bymass(M1,M2,F2,...,name='mixture name')
        """
        return cls(parts, by='mass', **kw)

    @classmethod
    def byvolume(cls, *parts, **kw):
        """
        Returns an alloy defined by relative volume of the constituents.

        Mixture.byvolume(M1,M2,F2,...,name='mixture name')
        """
        return cls(parts, by='volume', **kw)

    def __init__(self, parts, by='volume', name=None):
        # Split [M1,M2,F2,...] into [M1,M2,...], [F2,...]
        material = [parts[0]] + [parts[i] for i in range(1, len(parts),2)]
        fraction = [parts[i] for i in range(2, len(parts), 2)]
        # Convert M1,M2, ... to materials if necessary
        material = [p if isinstance(p,Material) else Material(p)
                     for p in material]
        self.material = material

        # Make the fractions into fittable parameters
        fraction = [Par.default(w,limits=(0,100), name=f.name+" count")
                    for w,f in zip(fraction,material)]
        self.fraction = fraction

        # Specify the volume calculator based on the type of fraction
        if by == 'volume':
            self._volume = _VolumeFraction(material)
        elif by == 'mass':
            self._volume = _MassFraction(material)
        else:
            raise ValueError('fraction must be one of volume, mass or number')

        if name is None: name = "+".join(p.name for p in material)
        self.name = name

    def parameters(self):
        """
        Adjustable parameters are the fractions associated with each
        constituent and the relative scale fraction used to tweak
        the overall density.
        """
        return dict(fraction=self.fraction,
                    material=[m.parameters() for m in self.material])

    def _density(self):
        """
        Compute the density of the mixture from the density and proportion
        of the individual components.
        """
        fraction = numpy.array([0.]+[m.value for m in self.fraction])
        fraction[0] = 100 - sum(fraction)
        if (fraction<0).any():
            return NaN
        volume = self._volume(fraction)
        density = numpy.array([m.density() for m in self.material])
        return numpy.sum(volume*density)
    density = property(_density,doc=_density.__doc__)

    def sld(self, probe):
        """
        Return the scattering length density and absorption of the mixture.
        """
        # Convert fractions into an array, with the final fraction
        fraction = numpy.hstack((0, [m.value for m in self.fraction]))
        fraction[0] = 100 - sum(fraction)
        if (fraction<0).any():
            return NaN, NaN

        # Lookup SLD
        slds = [c.sld(probe) for c in self.material]
        coh,absorp = [numpy.asarray(v) for v in zip(*slds)]

        # Use calculator to convert individual SLDs to overall SLD
        volume_fraction = self._volume(fraction)
        coh = numpy.sum(coh*volume_fraction)
        absorp = numpy.sum(absorp*volume_fraction[:,None],axis=0)
        #print "Mixture",self.name,coh,absorp

        return (coh,absorp)

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
    or the composition changes.  This can be done either by deleting
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
