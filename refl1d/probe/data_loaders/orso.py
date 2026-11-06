from pathlib import Path
from typing import Literal

import numpy as np
import yaml

import orsopy.fileio.model_language as orsopy_model
from orsopy.fileio.data_source import Sample as ORSOSample
from orsopy.fileio.model_language import Material as ORSOMaterial
from orsopy.fileio.orso import load_nexus, load_orso
from orsopy.utils.resolver_slddb import ResolverSLDDB

from periodictable.formulas import mix_by_volume, formula as pt_formula

from refl1d.probe.resolution import FWHM2sigma
from refl1d.sample.layers import Stack, Slab
from refl1d.sample.material import Mixture, BulkDensityMaterial, NumberDensityMaterial, SLD, Vacuum, Material


def load_orso_data(filename):
    """
    Load an ORSO text (.ort) or binary (.orb) file containing one or more datasets

    Parameters
    ----------
    filename : str
        The path to the ORSO file to be loaded.

    Returns
    -------
    list of tuple
        A list of tuples, each containing a header dictionary and a data array derived from each loaded dataset.
        The header dictionary contains metadata about the measurement,
        and the data array contains the measurement data.

    Notes
    -----
    The function supports both ORSO text (.ort) and binary (.orb) files.
    The polarization information is converted using a predefined mapping.
    The header dictionary includes keys for polarization, angle, angular resolution,
    wavelength, and wavelength resolution.
    """
    if filename.endswith(".ort"):
        entries = load_orso(filename)
    elif filename.endswith(".orb"):
        entries = load_nexus(filename)

    POL_CONVERSION = {
        "po": "++",
        "mo": "--",
        "mm": "--",
        "mp": "-+",
        "pm": "+-",
        "pp": "++",
    }

    entries_out = []
    for entry in entries:
        header = entry.info
        data = entry.data
        settings = header.data_source.measurement.instrument_settings
        columns = header.columns
        polarization = POL_CONVERSION.get(settings.polarization, "unpolarized")
        header_out = {"polarization": polarization}

        def get_key(orso_name, refl1d_name, refl1d_resolution_name):
            """
            Extract value and error from one of the ORSO columns. If no column corresponding
            to entry `orso_name` is found, search in the instrument settings.

            Parameters
            ----------
            orso_name : str
                The name of the ORSO column or instrument setting to extract.
            refl1d_name : str
                The corresponding refl1d name for the value of entry `orso_name`
            refl1d_resolution_name : str
                The corresponding refl1d error name the error of entry `orso_name`

            Notes
            -----
            This function requires the instrument setting `orso_name` to have a "magnitude" and "error" attribute.
            """
            column_index = next(
                (i for i, c in enumerate(columns) if getattr(c, "physical_quantity", None) == orso_name),
                None,
            )
            resolution_index = None
            if column_index is not None:
                # NOTE: this is based on column being second index (under debate in ORSO)
                header_out[refl1d_name] = data[:, column_index]
                cname = columns[column_index].name
                resolution_index, resolution_column = next(
                    ((i, c) for i, c in enumerate(columns) if getattr(c, "error_of", None) == cname),
                    (None, None),
                )
                if resolution_index is not None:
                    header_out[refl1d_resolution_name] = data[:, resolution_index]
                    if resolution_column.value_is == "FWHM":
                        header_out[refl1d_resolution_name] = FWHM2sigma(header_out[refl1d_resolution_name])

            # Fall back to instrument_settings if no column found
            v = getattr(settings, orso_name, None)
            if hasattr(v, "magnitude") and column_index is None:
                # only set if not already set from column
                header_out[refl1d_name] = v.magnitude
            if hasattr(v, "error") and resolution_index is None:
                header_out[refl1d_resolution_name] = v.error.error_value
                if v.error.value_is == "FWHM":
                    header_out[refl1d_resolution_name] = float(FWHM2sigma(header_out[refl1d_resolution_name]))

        get_key("incident_angle", "angle", "angular_resolution")
        get_key("wavelength", "wavelength", "wavelength_resolution")

        # convert error columns from FWHM to sigma if they are present
        # and need conversion
        if len(columns) >= 3:
            # third column is always uncertainty on reflectivity
            dR_col = columns[2]
            if dR_col.value_is == "FWHM":
                data[:, 2] = FWHM2sigma(data[:, 2])
        if len(columns) >= 4:
            # fourth column is always uncertainty on Q
            dQ_col = columns[3]
            if dQ_col.value_is == "FWHM":
                data[:, 3] = FWHM2sigma(data[:, 3])

        entries_out.append((header_out, np.array(data).T))
    return entries_out


def load_orso_sample(
    sample_file: Path | str,
    name: str | None = None,
    mode: Literal["sld", "formula", "mixture"] = "sld",
) -> Stack:
    converter = _OrsoConverter(sample_file, mode=mode)
    return converter.convert()


class _OrsoConverter:
    def __init__(self, filename, name=None, mode="sld"):
        self.mode = mode
        filename = Path(filename)
        name = name or filename.stem
        substacks = yaml.safe_load(Path(filename).read_text())
        self.sample = ORSOSample(name=name, model=substacks)
        self.materials = self.sample.model.materials

    def convert(self) -> Stack:
        orso_layers = self.sample.model.resolve_to_layers()
        refl1d_layers = [self.convert_layer(layer) for layer in orso_layers]

        return Stack(refl1d_layers)

    def convert_layer(self, layer: orsopy_model.Layer) -> Slab:
        refl1d_material = self.convert_material(layer.material)

        refl1d_layer = Slab(
            material=refl1d_material,
            thickness=layer.thickness.as_unit("angstrom"),
            interface=layer.roughness.as_unit("angstrom") if layer.roughness else None,
        )

        return refl1d_layer

    def convert_material(self, material: ORSOMaterial) -> Material:
        """
        material may be an sld, a formula + density or a composite

        density is mass_density or number_density or lookup by formula.

        If the density is not given then it may be guessed from the alloy
        composition. If all else fails the density is 0.0.

        relative_density is an additional scaling factor which applies to both
        sld and to formula.

        Exchangeable hydrogen use Hx in the formula.

        TODO: deuteration is not yet supported

        Composits are lists of materials, either from the materials section,
        or from looking up the formula in the ORSO database. For those from
        the database, we can lookup the cached density estimate, or the alloy
        density estimate if there is one.

        Depending on the translation mode, we may fit relative portions
        (mode: mixture), density (mode: formula) or just rho/irho (mode: sld)
        """
        # If material is defined by name in the orso database then parse
        # the comment field to extract the name, otherwise use the formula.
        # If no name and no formula, then it defaults to "SLD".
        if material.comment:
            # Autogenerated comment is something like:
            #    material 'DMPC' from ORSO SLD db ID=296
            if material.comment.startswith("material '"):
                name = material.comment.split("'")[1]
            else:
                name = material.comment
        else:
            name = getattr(material, "formula", None)

        # Use the name given by the user from the materials and composits lists
        # Database lookups are not in the materials list which is why we need
        # the code above.
        for k, v in {**self.materials, **self.sample.model.composits}.items():
            if id(v) == id(material):
                name = k
                break

        if isinstance(material, orsopy_model.Composit):
            return self.process_composit(material, name)

        relative_density = material.relative_density or 1.0

        # Resolve the sld using orsopy calculator
        if self.mode == "sld" or (material.mass_density is None and material.number_density is None):
            sld_value = relative_density * material.get_sld() * 1e6  # in 1e-6 A^-2
            return SLD(rho=sld_value.real, irho=sld_value.imag, name=name)

        # Parse the formula using periodictable parser, with Hx -> H[1] substitution
        formula = parse_formula(material.formula)
        if material.mass_density is not None:
            density = material.mass_density.as_unit("g/cm^3")
        else:  # material.number_density is not None:
            # Note: converting number density to mass density rather than using
            # the NumberDensityMaterial class.
            number_density = material.number_density.as_unit("1/cm^3")
            density = number_density * formula.molecular_mass
        return BulkDensityMaterial(formula=formula, density=relative_density * density, name=name)

    def process_composit(self, material, name=None):
        # Note: using bb.CACHED_MATERIALS rather than importing in case it gets
        # reset or otherwise changes out from under us.
        import orsopy.fileio.model_building_blocks as bb

        # Dead code: composits are named
        if name is None:
            name = "+".join(material.composition.keys())

        # If we have a mixture of formulas and slds then we will be forced
        # to use sld everywhere. Otherwise we can convert number density
        # to mass density and make a mixture of number density materials.

        # resolve components to materials and check if they all have formulas
        parts = []
        has_formula = True
        for component, fraction in material.composition.items():
            if component in self.materials:
                material = self.materials[component]
            else:
                # Get the cached number density for materials from the database
                number_density = bb.CACHED_MATERIALS[component][0]
                material = ORSOMaterial(formula=component, number_density=number_density)
            has_formula = has_formula and (material.formula is not None)
            parts.append((component, fraction, material))

        # Simple case: use ORSO to lookup sld
        if self.mode == "sld" or not has_formula:
            sld = 0.0
            for component, fraction, material in parts:
                sld += material.get_sld() * fraction * 1e6
            return SLD(rho=sld.real, irho=sld.imag, name=name)

        # For the other cases we need the mass density
        converted = []
        for component, fraction, material in parts:
            relative_density = material.relative_density or 1.0
            formula = parse_formula(material.formula)
            if material.mass_density is not None:
                density = material.mass_density.as_unit("g/cm^3")
            else:
                number_density = material.number_density.as_unit("1/cm^3")
                density = number_density * formula.molecular_mass
            formula.density = density * relative_density
            converted.append((component, formula, fraction))

        if self.mode == "formula":
            fragments = []
            for component, formula, fraction in converted:
                fragments.extend((formula, fraction))
            return BulkDensityMaterial(mix_by_volume(*fragments), name=name)

        if self.mode == "mixture":
            fragments = []
            for component, formula, fraction in converted:
                fragment = BulkDensityMaterial(formula, name=component)
                fragments.extend((fragment, fraction * 100))
            return Mixture(base=fragments[0], parts=fragments[2:], name=name)

        raise ValueError(f"Unrecognized mode={self.mode}")


def parse_formula(f):
    # Hx => H[1] for exchangeable hydrogen
    return pt_formula(f.replace("Hx", "H[1]"))
