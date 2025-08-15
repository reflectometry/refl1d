import numpy as np
from orsopy.fileio.orso import load_nexus, load_orso
import orsopy.fileio.model_language as orsopy_model
from orsopy.fileio.model_language import Layer as ORSOLayer, SampleModel as ORSOSample, Material as ORSOMaterial
from orsopy.utils.resolver_slddb import ResolverSLDDB
from refl1d.sample.layers import Stack, Slab
from refl1d.sample.material import Compound, Mixture, BulkDensityMaterial, NumberDensityMaterial, SLD, Vacuum


def parse_orso(filename):
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
            if column_index is not None:
                # NOTE: this is based on column being second index (under debate in ORSO)
                header_out[refl1d_name] = data[:, column_index]
                cname = columns[column_index].name
                resolution_index = next(
                    (i for i, c in enumerate(columns) if getattr(c, "error_of", None) == cname),
                    None,
                )
                if resolution_index is not None:
                    header_out[refl1d_resolution_name] = data[:, resolution_index]
            else:
                v = getattr(settings, orso_name, None)
                if hasattr(v, "magnitude"):
                    header_out[refl1d_name] = v.magnitude
                if hasattr(v, "error"):
                    header_out[refl1d_resolution_name] = v.error.error_value

        get_key("incident_angle", "angle", "angular_resolution")
        get_key("wavelength", "wavelength", "wavelength_resolution")

        entries_out.append((header_out, np.array(data).T))
    return entries_out


def orso_sample_converter(model: orsopy_model.SampleModel):
    """
    Convert an ORSO sample model to a refl1d Stack model.

    Parameters
    ----------
    model : ORSOSample
        The ORSO sample model to convert.

    Returns
    -------
    refl1d.sample.layers.Stack
        The converted refl1d model.
    """

    orso_layers = model.resolve_to_layers()

    refl1d_layers = [orso_layer_converter(layer) for layer in orso_layers]

    return Stack(refl1d_layers)


def orso_layer_converter(layer: orsopy_model.Layer):
    """
    Convert an ORSO layer to a refl1d Slab.

    Parameters
    ----------
    layer : ORSOSample.Layer
        The ORSO layer to convert.

    Returns
    -------
    refl1d.sample.layers.Slab
        The converted refl1d slab.
    """

    refl1d_material = orso_material_converter(layer.material)

    refl1d_layer = Slab(
        material=refl1d_material,
        thickness=layer.thickness.as_unit("angstrom"),
        interface=layer.roughness.as_unit("angstrom") if layer.roughness else None,
    )

    return refl1d_layer


def orso_material_converter(material: ORSOMaterial):
    """
    Convert an ORSO material to a refl1d Material.

    Parameters
    ----------
    material : ORSOMaterial
        The ORSO material to convert.

    Returns
    -------
    refl1d.sample.material.Material
        The converted refl1d material.
    """
    if isinstance(material, orsopy_model.Composit):
        parts = []
        for component, fraction in material.composition.items():
            # TODO: how are we supposed to get the number density from ORSO?
            number_density = ResolverSLDDB().resolve_formula(component)  #  in 1/nm³
            cmaterial = NumberDensityMaterial(formula=component, number_density=number_density)
            parts.extend([cmaterial, fraction])
        # Mixture is expecting a list [base, M2, F2, M3, F3, ...]
        # but ORSO Composit does not have a base material,
        # so we will set that to vacuum with fraction 0.0 (implicitly)
        # as the other fractions add up to 1.0
        return Mixture(
            base=Vacuum(),
            parts=parts,
        )
    elif material.mass_density is not None:
        return BulkDensityMaterial(formula=material.formula, density=material.mass_density.as_unit("g/cm^3"))
    elif material.number_density is not None:
        return NumberDensityMaterial(formula=material.formula, number_density=material.number_density.as_unit("1/cm^3"))
    elif material.sld is not None:
        sld_value = material.sld.as_unit("1/angstrom^2") * 1e6  # in 1e-6 A^-2
        return SLD(rho=sld_value.real, irho=sld_value.imag)
    else:
        raise ValueError(f"Unsupported material: {material}")
