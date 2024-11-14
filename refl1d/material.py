"""
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

__all__ = [
    "Material",
    "Mixture",
    "SLD",
    "Vacuum",
    "Scatterer",
    "ProbeCache",
    "BulkDensityMaterial",
    "NaturalDensityMaterial",
    "NumberDensityMaterial",
    "RelativeDensityMaterial",
    "CellVolumeMaterial",
]

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import periodictable
from bumps.parameter import Expression, Parameter
from numpy import inf, nan
from periodictable.constants import avogadro_number
from periodictable.formulas import Formula as BaseFormula


class Scatterer:
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

    def sld(self, sf) -> Tuple[float, float]:
        """
        Return the scattering length density pair (rho, irho) expected for the given
        scattering factors, as returned from a call to scattering_factors()
        for a particular probe.
        """
        raise NotImplementedError()

    def __or__(self, other):
        """
        Interface for a material stacker, to support e.g., *stack = Si | air*.
        """
        raise NotImplementedError("failed monkey-patch: material stacker needs" " to replace __or__ in Scatterer")

    def __call__(self, *args, **kw):
        """
        Interface for a material stacker, to support e.g., *stack = Si(thickness=)*.
        """
        raise NotImplementedError("failed monkey-patch: material stacker needs" " to replace __call__ in Scatterer")

    def __str__(self):
        return self.name


# ============================ No scatterer =============================
@dataclass(init=False)
class Vacuum(Scatterer):
    """
    Empty layer (all sld = 0)
    """

    def __init__(self, *args, **kw):
        self.name = "Vacuum"

    def parameters(self):
        return []

    def to_dict(self):
        return {
            "type": type(self).__name__,
        }

    def sld(self, probe):
        return 0, 0

    def __repr__(self):
        return "Vacuum()"


# ============================ Unknown scatterer ========================
@dataclass(init=False)
class SLD(Scatterer):
    r"""
    Unknown composition.

    Use this when you don't know the composition of the sample.  The
    absorption and scattering length density are stored directly rather
    than trying to guess at the composition from details about the sample.

    The complex scattering potential is defined by $\rho + j \rho_i$.
    Note that this differs from $\rho + j \mu/(2 \lambda)$ more
    traditionally used in neutron reflectometry, and $N r_e (f_1 + j f_2)$
    traditionally used in X-ray reflectometry.

    Given that $f_1$ and $f_2$ are always wavelength dependent for X-ray
    reflectometry, it will not make much sense to uses this for wavelength
    varying X-ray measurements.  Similarly, some isotopes, particularly
    rare earths, show wavelength dependence for neutrons, and so
    time-of-flight measurements should not be fit with a fixed SLD scatterer.
    """

    name: str
    rho: Parameter
    irho: Parameter

    def __init__(self, name="SLD", rho: Union[float, Parameter] = 0, irho=0):
        self.name = name
        self.rho = Parameter.default(rho, name=name + " rho")
        self.irho = Parameter.default(irho, name=name + " irho")

    def parameters(self):
        return {"rho": self.rho, "irho": self.irho}

    def sld(self, probe):
        return self.rho.value, self.irho.value


# ============================ Substances =============================
# TODO: fix and use the real (circular) type definition:
# FormulaStructure = Tuple['FormulaUnit']
# FormulaUnit = Tuple[float, Union[str, FormulaStructure]]
# still have to convert Element to/from string repr
# ELEMENTS = Enum('elements', [(e.name, e.symbol) for e in periodictable.elements._element.values()])


class BaseMaterial(Scatterer):
    """
    Description of a solid block of material.

    :Parameters:
        *name* : string

        *formula* : Formula

            Composition can be initialized from either a string or a chemical
            formula.  Valid values are defined in periodictable.formula.

        *use_incoherent* = False : boolean

            True if incoherent scattering should be interpreted as absorption.

    """

    name: str
    formula: str  # Formula
    use_incoherent: bool = False

    # not in the schema:
    _formula: BaseFormula

    def __init__(
        self, formula: Union[str, BaseFormula], density=None, natural_density=None, name=None, use_incoherent=False
    ):
        # coerce formula to string repr:
        self.formula = str(formula)
        self._formula: BaseFormula = periodictable.formula(formula, density=density, natural_density=natural_density)
        self.name = name if name is not None else str(self._formula)
        self.use_incoherent = use_incoherent

    def sld(self, probe):
        rho, irho, incoh = probe.scattering_factors(self._formula, density=self.density.value)
        if self.use_incoherent:
            raise NotImplementedError("incoherent scattering not supported")
            # irho += incoh
        return rho, irho

    def __str__(self):
        return self.name


FitByChoices = Literal["bulk_density", "natural_density", "relative_density", "number_density", "cell_volume"]


def Material(
    formula: Union[str, BaseFormula],
    density: Optional[float] = None,
    natural_density: Optional[float] = None,
    name: Optional[str] = None,
    use_incoherent: bool = False,
    fitby: FitByChoices = "bulk_density",
    **kw,
):
    if fitby == "bulk_density":
        return BulkDensityMaterial(formula, density=density, name=name, use_incoherent=use_incoherent)
    elif fitby == "natural_density":
        return NaturalDensityMaterial(
            formula, natural_density=natural_density, name=name, use_incoherent=use_incoherent
        )
    elif fitby == "relative_density":
        return RelativeDensityMaterial(
            formula, density=density, natural_density=natural_density, name=name, use_incoherent=use_incoherent, **kw
        )
    elif fitby == "number_density":
        return NumberDensityMaterial(
            formula, density=density, natural_density=natural_density, name=name, use_incoherent=use_incoherent, **kw
        )
    elif fitby == "cell_volume":
        return CellVolumeMaterial(
            formula, density=density, natural_density=natural_density, name=name, use_incoherent=use_incoherent, **kw
        )
    else:
        raise ValueError(
            f'unknown value "{fitby}" of fitby: should be one of ["bulk_density", "natural_density", "relative_density", "number_density", "cell_volume"]'
        )


@dataclass(init=False)
class BulkDensityMaterial(BaseMaterial):
    """
    A solid block of material, described by its bulk density

    :Parameters:
        *density* : float | |g/cm^3|
            the bulk density for the material.
    """

    name: str
    formula: str  # Formula
    use_incoherent: bool = False
    density: Parameter

    def __init__(
        self,
        formula: Union[str, BaseFormula],
        density: Optional[Union[float, Parameter]] = None,
        name=None,
        use_incoherent=False,
    ):
        BaseMaterial.__init__(self, formula, density=density, name=name, use_incoherent=use_incoherent)
        if density is None:
            if self._formula.density is not None:
                density = self._formula.density
            else:
                raise ValueError(
                    f"material {self._formula} does not have known density: please provide it in arguments"
                )
        self.density = Parameter.default(density, name=self.name + " density", limits=(0, inf))
        self.bulk_density = self.density

    def parameters(self):
        return dict(density=self.density)


@dataclass(init=False)
class NaturalDensityMaterial(BaseMaterial):
    """
    A solid block of material, described by its natural density

    :Parameters:
        *natural_density* : float | |g/cm^3|
            the natural bulk density for the material.
    """

    name: str
    formula: str  # Formula
    use_incoherent: bool = False
    natural_density: Parameter
    density: Expression

    def __init__(
        self,
        formula: Union[str, BaseFormula],
        natural_density: Optional[Union[float, Parameter]] = None,
        name=None,
        use_incoherent=False,
    ):
        BaseMaterial.__init__(self, formula, natural_density=natural_density, name=name, use_incoherent=use_incoherent)
        if natural_density is None:
            if self._formula.density is not None:
                natural_density = self._formula.density
            else:
                raise ValueError(
                    f"material {self._formula} does not have known natural_density: please provide it in arguments"
                )
        self.natural_density = Parameter.default(natural_density, name=self.name + " nat. density", limits=(0, inf))
        self.density = self.natural_density / self._formula.natural_mass_ratio()

    def parameters(self):
        return dict(density=self.density, natural_density=self.natural_density)


@dataclass(init=False)
class NumberDensityMaterial(BaseMaterial):
    """
    A solid block of material, described by its number density

    :Parameters:
        *number_density*: [atoms/cm^3]
                Density is *number_density* * molar mass / avogadro constant

        *density* : float | |g/cm^3|
            if specified, the bulk density for the material.

        *natural_density* : float | |g/cm^3|
            if specified, the natural bulk density for the material.
    """

    name: str
    formula: str  # Formula
    use_incoherent: bool = False
    number_density: Parameter
    density: Expression

    def __init__(
        self,
        formula: Union[str, BaseFormula],
        number_density: Optional[Union[float, Parameter]] = None,
        density: Optional[float] = None,
        natural_density: Optional[float] = None,
        name=None,
        use_incoherent=False,
    ):
        BaseMaterial.__init__(
            self, formula, density=density, natural_density=natural_density, name=name, use_incoherent=use_incoherent
        )
        if number_density is None:
            if self._formula.density is not None:
                number_density = avogadro_number / self._formula.mass * self._formula.density
            else:
                raise ValueError(
                    f"material {self._formula} does not have known density: please provide density or natural_density argument"
                )
        self.number_density = Parameter.default(number_density, name=self.name + " number density", limits=(0, inf))
        self.density = self.number_density / avogadro_number * self._formula.mass
        self._formula.density = float(self.density)

    def parameters(self):
        return dict(density=self.density, number_density=self.number_density)


@dataclass(init=False)
class RelativeDensityMaterial(BaseMaterial):
    """
    A solid block of material, described by its relative density

    :Parameters:
        *relative_density* : unitless
            Density is *relative_density* * formula density

        *density* : float | |g/cm^3|
            if specified, the bulk density for the material.

        *natural_density* : float | |g/cm^3|
            if specified, the natural bulk density for the material.
    """

    name: str
    formula: str  # Formula
    use_incoherent: bool = False
    relative_density: Parameter
    density: Expression

    def __init__(
        self,
        formula: Union[str, BaseFormula],
        relative_density: Optional[Union[float, Parameter]] = None,
        density: Optional[float] = None,
        natural_density: Optional[float] = None,
        name=None,
        use_incoherent=False,
    ):
        BaseMaterial.__init__(
            self, formula, density=density, natural_density=natural_density, name=name, use_incoherent=use_incoherent
        )
        if self._formula.density is None and density is None and natural_density is None:
            raise ValueError(
                f"material {self._formula} does not have known density: please provide density or natural_density argument"
            )
        if relative_density is None:
            relative_density = 1
        self.relative_density = Parameter.default(relative_density, name=self.name + " rel. density", limits=(0, inf))
        self.density = self._formula.density * self.relative_density

    def parameters(self):
        return dict(density=self.density, relative_density=self.relative_density)


@dataclass(init=False)
class CellVolumeMaterial(BaseMaterial):
    """
    A solid block of material, described by the volume of one unit cell

    :Parameters:
        *cell_volume* : |Ang^3|
                Density is mass / *cell_volume*

        *density* : float | |g/cm^3|
            if specified, the bulk density for the material.

        *natural_density* : float | |g/cm^3|
            if specified, the natural bulk density for the material.


    For example, to fit Pd by cell volume use::
        >>> m = Material('Pd', fitby='cell_volume')
        >>> m.cell_volume.range(1, 10)
        Parameter(Pd cell volume)
        >>> print("%.2f %.2f"%(m.density.value, m.cell_volume.value))
        12.02 14.70

    """

    name: str
    formula: str  # Formula
    use_incoherent: bool = False
    cell_volume: Parameter
    density: Expression

    def __init__(
        self,
        formula: Union[str, BaseFormula],
        cell_volume: Optional[Union[float, Parameter]] = None,
        density: Optional[float] = None,
        natural_density: Optional[float] = None,
        name=None,
        use_incoherent=False,
    ):
        BaseMaterial.__init__(
            self, formula, density=density, natural_density=natural_density, name=name, use_incoherent=use_incoherent
        )
        if self._formula.density is None and density is None and natural_density is None:
            raise ValueError(
                f"material {self._formula} does not have known density: please provide density or natural_density argument"
            )
        if cell_volume is None:
            cell_volume = (1e24 * self._formula.molecular_mass) / self._formula.density
        self.cell_volume = Parameter.default(cell_volume, name=self.name + " cell volume", limits=(0, inf))
        self.density = (1e24 * self._formula.molecular_mass) / self.cell_volume

    def parameters(self):
        return dict(density=self.density, cell_volume=self.cell_volume)


MaterialTypes = Union[
    BulkDensityMaterial, NaturalDensityMaterial, NumberDensityMaterial, RelativeDensityMaterial, CellVolumeMaterial
]


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

    def __init__(self, parts=None):
        # Split [M1, N1, M2, N2, ...] into [M1, M2, ...], [N1, N2, ...]
        formula = [parts[i] for i in range(0, len(parts), 2)]
        count = [parts[i] for i in range(1, len(parts), 2)]
        # Convert M1, M2, ... to materials if necessary
        formula = [periodictable.formula(p) for p in formula]
        count = [Parameter.default(w, limits=(0, None), name=str(f) + " count") for w, f in zip(count, formula)]
        self.parts = formula
        self.count = count

    def parameters(self):
        """
        Adjustable parameters are the fractions associated with each
        constituent and the relative scale fraction used to tweak
        the overall density.
        """
        return {"count": self.count}

    def formula(self):
        return tuple((c.value, f) for c, f in zip(self.count, self.parts))

    def __str__(self):
        return "<%s>" % (", ".join(str(M) for M in self.formula()))

    def __repr__(self):
        return "Compound([%s])" % (", ".join(repr(M) for M in self.formula()))


# ============================ Alloys =============================


class _VolumeFraction(object):
    """
    Returns the relative volume for each component in the system given
    the volume percentages.
    """

    def __init__(self, base, material):
        pass

    def __call__(self, fraction):
        return 0.01 * np.asarray(fraction)


class _MassFraction(object):
    """
    Returns the relative volume for each component in the system given
    the relative masses.
    """

    def __init__(self, base, material):
        self._material = [base] + material

    def __call__(self, fraction):
        density = np.array([m.density.value for m in self._material])
        volume = fraction / density
        return volume / sum(volume)


class Mixture(Scatterer):
    """
    Mixed block of material.

    The components of the mixture can vary relative to each other, either
    by mass, by volume or by number::

        Mixture.bymass(base, M2, F2..., name='mixture name')
        Mixture.byvolume(base, M2, F2..., name='mixture name')

    The materials *base*, *M2*, *M3*, ... can be chemical formula
    strings including @density or from material objects. Use natural_density
    to change from bulk values if the formula has isotope substitutions.

    The fractions F2, F3, ... are percentages in [0, 100]. The implicit
    fraction for the base material is 100 - (F2+F3+...). The SLD is NaN
    then *F1 < 0*.

    name defaults to M2.name+M3.name+...
    """

    @classmethod
    def bymass(cls, base, *parts, **kw):
        """
        Returns an alloy defined by relative mass of the constituents.

        Mixture.bymass(base, M2, F2, ..., name='mixture name')
        """
        return cls(base, parts, by="mass", **kw)

    @classmethod
    def byvolume(cls, base, *parts, **kw):
        """
        Returns an alloy defined by relative volume of the constituents.

        Mixture.byvolume(base, M2, F2, ..., name='mixture name')
        """
        return cls(base, parts, by="volume", **kw)

    def __init__(self, base, parts, by="volume", name=None, use_incoherent=False):
        # Split [M1, M2, F2, ...] into [M1, M2, ...], [F2, ...]
        material = [parts[i] for i in range(0, len(parts), 2)]
        fraction = [parts[i] for i in range(1, len(parts), 2)]
        # Convert M1, M2, ... to materials if necessary
        if not isinstance(base, Scatterer):
            base = Material(base)
        material = [p if isinstance(p, Scatterer) else Material(p) for p in material]

        # Specify the volume calculator based on the type of fraction
        if by == "volume":
            _volume = _VolumeFraction(base, material)
        elif by == "mass":
            _volume = _MassFraction(base, material)
        else:
            raise ValueError("fraction must be one of volume, mass or number")

        # Name defaults to names of individual components
        if name is None:
            name = "+".join(p.name for p in material)

        # Make the fractions into fittable parameters
        fraction = [Parameter.default(w, limits=(0, 100), name=f.name + "% " + by) for w, f in zip(fraction, material)]

        self._volume = _volume
        self.base = base
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
        return {
            "base": self.base.parameters(),
            "material": [m.parameters() for m in self.material],
            "fraction": self.fraction,
        }

    def _density(self):
        """
        Compute the density of the mixture from the density and proportion
        of the individual components.
        """
        fraction = np.array([0.0] + [m.value for m in self.fraction])
        # TODO: handle invalid fractions using penalty functions
        # S = sum(fraction)
        # scale = S/100 if S > 100 else 1
        # fraction[0] = 100 - S/scale
        # penalty = scale - 1
        fraction[0] = 100 - sum(fraction)
        if (fraction < 0).any():
            return nan
        volume = self._volume(fraction)
        density = np.array([m.density() for m in [self.base] + self.material])
        return np.sum(volume * density)

    density = property(_density, doc=_density.__doc__)

    def sld(self, probe):
        """
        Return the scattering length density and absorption of the mixture.
        """
        # Convert fractions into an array, with the final fraction
        fraction = np.array([0.0] + [f.value for f in self.fraction])
        fraction[0] = 100 - sum(fraction)
        # TODO: handle invalid fractions using penalty functions
        # S = sum(fraction)
        # scale = S/100 if S > 100 else 1
        # fraction[0] = 100 - S/scale
        # penalty = scale - 1
        if (fraction < 0).any():
            return nan, nan

        # Lookup SLD
        slds = [c.sld(probe) for c in [self.base] + self.material]
        rho, irho = [np.asarray(v) for v in zip(*slds)]

        # Use calculator to convert individual SLDs to overall SLD
        volume_fraction = self._volume(fraction)
        rho = np.sum(rho * extend(volume_fraction, rho))

        irho = np.sum(irho * extend(volume_fraction, irho))
        if self.use_incoherent:
            raise NotImplementedError("incoherent scattering not supported")
        # print "Mixture", self.name, coh, absorp

        return rho, irho

    def __str__(self):
        return "<%s>" % (", ".join(str(M) for M in [self.base] + self.material))

    def __repr__(self):
        return "Mixture(%s)" % (", ".join(repr(M) for M in [self.base] + self.material))


# ============================ SLD cache =============================


class ProbeCache(object):
    """
    Probe proxy for materials properties.

    A caching probe which only looks up scattering factors for materials
    which it hasn't seen before.   Note that caching is based on object
    id, and will fail if the material object is updated with a new atomic
    structure.

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

    def scattering_factors(self, material, density):
        """
        Return the scattering factors for the material, retrieving them from
        the cache if they have already been looked up.
        """
        h = id(material)
        if h not in self._cache:
            # lookup density of 1, and scale to actual density on retrieval
            self._cache[h] = self._probe.scattering_factors(material, density=1.0)
        return [v * density for v in self._cache[h]]


def extend(a, b):
    """
    Extend *a* to match the number of dimensions of *b*.

    This adds dimensions to the end of *a* rather than the beginning. It is
    equivalent to *a[..., None, None]* with the right number of None elements
    to make the number of dimensions match (or np.newaxis if you prefer).

    For example::

        from numpy.random import rand
        a, b = rand(3, 4), rand(3, 4, 2)
        a + b
        ==> ValueError: operands could not be broadcast together with shapes (3,4) (3,4,2)
        c = extend(a, b) + b
        c.shape
        ==> (3, 4, 2)

    Numpy broadcasting rules automatically extend arrays to the beginning,
    so the corresponding *lextend* function is not needed::

        c = rand(3, 4) + rand(2, 3, 4)
        c.shape
        ==> (2, 3, 4)
    """
    if np.isscalar(a):
        return a
    # CRUFT: python 2.7 support
    extra_dims = (1,) * (b.ndim - a.ndim)
    return a.reshape(a.shape + extra_dims)
    # python 3 uses
    # extra_dims = (np.newaxis,)*(b.ndim-a.ndim)
    # return a[(..., *extra_dims)]
