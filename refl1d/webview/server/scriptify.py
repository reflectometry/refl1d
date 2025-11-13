from dataclasses import asdict
import json
from textwrap import dedent
from typing import Tuple, Union
from refl1d.experiment import Experiment, FitProblem, MixedExperiment, Parameter, QProbe
from refl1d.probe.probe import BaseProbe, Probe, PolarizedNeutronProbe, QProbe, ProbeSet
from refl1d.sample.material import SLD
import refl1d.sample.magnetism as mag
import refl1d.sample.material as mat
from refl1d.sample.layers import Layer, Slab, Stack


def serialize_slab(slab: Slab) -> str:
    """
    Serialize a Slab instance to a string representation.

    Parameters:
        slab (Slab): The Slab instance to serialize.

    Returns:
        str: A string representation of the Slab instance.
    """
    if isinstance(slab, Slab):
        s_material = serialize_material(slab.material)
        s_magnetism = serialize_magnetism(slab.magnetism)
        s_slab = f'Slab(name="{slab.name}", thickness={slab.thickness.value}, interface={slab.interface.value}, material={s_material}, magnetism={s_magnetism})'
    else:
        raise NotImplementedError("Only Slab is supported.")
    return s_slab


def serialize_material(material: Union[mat.SLD, mat.MaterialTypes, mat.Vacuum]) -> str:
    """
    Serialize a material to a string representation.

    Parameters:
        material (SLD): The SLD instance to serialize.

    Returns:
        str: A string representation of the SLD instance.
    """
    if isinstance(material, mat.SLD):
        return serialize_sld(material)
    else:
        raise NotImplementedError("Only SLD materials are supported.")


def serialize_sld(sld: SLD) -> str:
    """
    Serialize a SLD instance to a string representation.

    Parameters:
        sld (SLD): The SLD instance to serialize.

    Returns:
        str: A string representation of the SLD instance.
    """
    return f'SLD(name="{sld.name}", rho={sld.rho.value}, irho={sld.irho.value})'


def serialize_magnetism(magnetism: mag.Magnetism) -> str:
    """
    Serialize a Magnetism instance to a string representation.

    Parameters:
        magnetism (Magnetism): The Magnetism instance to serialize.

    Returns:
        str: A string representation of the Magnetism instance.
    """
    if magnetism is None:
        return "None"
    elif isinstance(magnetism, mag.Magnetism):
        return f"Magnetism(name={magnetism.name}, rhoM={magnetism.rhoM.value}, thetaM={magnetism.thetaM.value})"
    else:
        raise NotImplementedError("Only Magnetism and None are supported.")


def serialize_sample(sample: Stack, counter: int = None) -> Tuple[str, str]:
    """
    Serialize a Stack instance to a string representation.

    Parameters:
        sample (Stack): The Stack instance to serialize.

    Returns:
        str: A string representation of the Stack instance.
    """
    counter_string = f"_{counter}" if counter is not None else ""
    sample_name = f"sample{counter_string}"
    slabs_name = f"slabs{counter_string}"
    if isinstance(sample, Stack):
        s_slabs = []
        s_thickness = []
        s_interface = []
        s_rho = []
        s_irho = []
        s_rhoM = []
        s_thetaM = []
        for i, slab in enumerate(sample):
            slab_name = f"{slabs_name}[{i}]"
            s_slab = serialize_slab(slab)
            s_slabs.append(s_slab)
            s_thickness.append(set_fitrange(slab.thickness, f"{slab_name}.thickness"))
            s_interface.append(set_fitrange(slab.interface, f"{slab_name}.interface"))
            s_rho.append(set_fitrange(slab.material.rho, f"{slab_name}.material.rho"))
            s_irho.append(set_fitrange(slab.material.irho, f"{slab_name}.material.irho"))
            if slab.magnetism is not None:
                s_rhoM.append(set_fitrange(slab.magnetism.rhoM, f"{slab_name}.magnetism.rhoM"))
                s_thetaM.append(set_fitrange(slab.magnetism.thetaM, f"{slab_name}.magnetism.thetaM"))

        s_parameters = [
            "",
            "#####################",
            "# Sample Parameters #",
            "#####################",
            "",
            *s_thickness,
            "",
            *s_interface,
            "",
            *s_rho,
            "",
            *s_irho,
            "",
            *s_rhoM,
            "",
            *s_thetaM,
            "",
        ]
        s_parameter = "\n".join(s_parameters)

        # Create the separator outside the f-string to avoid backslash in f-string expression (Python 3.10 compatibility)
        separator = ",\n    "
        s_stack = f"{slabs_name} = [\n    {separator.join(s_slabs)},\n]"
        return f"{s_stack}\n\n{sample_name} = Stack({slabs_name})\n{s_parameter}", sample_name
    else:
        raise NotImplementedError("Only Stack is supported.")


def set_fitrange(param: Parameter, path: str):
    """
    Set the fit range for a parameter.

    Parameters:
        param (Parameter): The parameter to set the fit range for.
        path (str): The path to the parameter.

    Returns:
        str: A string representation of the fit range.
    """
    if param.fittable and not param.fixed:
        return f"{path}.range({param.bounds[0]}, {param.bounds[1]})"
    else:
        return f"# {path}.range({param.bounds[0]}, {param.bounds[1]})"


def serialize_experiment(experiment: Experiment, counter: int = None) -> Tuple[str, str]:
    """
    Serialize an Experiment instance to a string representation.

    Parameters:
        experiment (Experiment): The Experiment instance to serialize.

    Returns:
        str: A string representation of the Experiment instance.
    """
    counter_string = f"_{counter}" if counter is not None else ""

    DOCSTRING = dedent("""\
        # === Experiment Arguments ===
        # a model (Experiment) object is initialized with these arguments:
        # with the following arguments:
        #  - sample (required) is a Stack object that contains the layers of the sample
        #  - probe  (required) is a Probe object that contains the probe information
        #  - dz is the step size in Angstroms to be used for rendering the profile
        #  - step_interfaces is a boolean that determines whether to use the
        #    Nevot-Croce approximation for roughness (True) or to use the
        #    microslabbed interfaces (False)
        #  - dA is an aggregation constant for combining smooth regions of the profile
        #    prior to calculating R (greatly speeds up fits)
    """)
    experiment_name = f"experiment{counter_string}"
    if isinstance(experiment, Experiment):
        s_sample, sample_name = serialize_sample(experiment.sample, counter=counter)
        probe_name = f"probe{counter_string}"
        serialized_probe = serialize_probe(experiment.probe)
        # serialized_probe = json.dumps( bumps.serialize.serialize(experiment.probe) )
        s_probe = f"{probe_name} = {serialized_probe}\n"  # Add specific probe details here"
        s_header = comment_header(f"Experiment{counter_string}")
        s_experiment = f"{experiment_name} = Experiment(sample={sample_name}, probe={probe_name}, dz=1.0, step_interfaces=False, dA=1.0)"
        return f"{s_header}\n\n{s_probe}\n{s_sample}\n{DOCSTRING}\n{s_experiment}", experiment_name
    else:
        raise NotImplementedError("Only Experiment is supported.")


def serialize_fitproblem(problem: FitProblem[Experiment]) -> str:
    """
    Serialize a FitProblem instance to a string representation.

    Parameters:
        problem (FitProblem): The FitProblem instance to serialize.

    Returns:
        str: A string representation of the FitProblem instance.
    """
    experiments = list(problem.models)
    s_experiments = []
    s_experiment_names = []
    for i, experiment in enumerate(experiments, start=1):
        s_experiment, s_experiment_name = serialize_experiment(experiment, counter=i)
        s_experiments.append(s_experiment)
        s_experiment_names.append(s_experiment_name)

    s_experiment = "\n\n".join(s_experiments)
    s_experiment_names = ", ".join(s_experiment_names)

    s_imports = "from refl1d.names import *\n\n"
    s_header = comment_header("FitProblem")
    s_problem = f"{s_imports}{s_experiment}\n\n{s_header}\nproblem = FitProblem([{s_experiment_names}])"

    return s_problem


def comment_header(header: str) -> str:
    """
    Generate a comment header for the script.

    Parameters:
        header (str): The header text.

    Returns:
        str: A string representation of the comment header.
    """
    width = len(header) + 4
    return f"{'#' * width}\n# {header} #\n{'#' * width}\n"


def serialize_probe(probe: BaseProbe) -> Tuple[str, str]:
    """
    Serialize a BaseProbe instance to a string representation.

    Parameters:
        probe (BaseProbe): The BaseProbe instance to serialize.

    Returns:
        str: A string representation of the BaseProbe instance.
    """
    if hasattr(probe, "filename"):
        filename = probe.filename
    elif hasattr(probe, "xs"):
        for xs in probe.xs:
            if getattr(xs, "filename", None) is not None:
                filename = xs.filename
                break
    print("filename: ", filename)
    if filename is not None:
        return f'load4("{filename}")'

    if isinstance(probe, ProbeSet):
        raise NotImplementedError("ProbeSet is not supported.")
    elif isinstance(probe, Probe):
        return serialize_plain_probe(probe)
    elif isinstance(probe, QProbe):
        return serialize_qprobe(probe)
    else:
        raise NotImplementedError("Only Probe is supported.")


def serialize_plain_probe(probe: Probe):
    """
    Serialize a Probe instance to a string representation.

    Parameters:
        probe (Probe): The Probe instance to serialize.

    Returns:
        str: A string representation of the Probe instance.

    Probe attributes:

    intensity: Parameter
    background: Parameter
    back_absorption: Parameter
    theta_offset: Parameter
    sample_broadening: Parameter
    name: Optional[str] = None
    filename: Optional[str] = None
    back_reflectivity: bool = False
    R: Optional[Any] = None
    dR: Optional[Any] = 0
    T: "NDArray" = field_desc("List of theta values (incident angle)")
    dT: Optional[Any] = 0
    L: "NDArray" = field_desc("List of lambda values (wavelength, in Angstroms)")
    dL: Optional[Any] = 0
    dQo: Optional[Union[Sequence, "NDArray"]] = None
    resolution: Literal["normal", "uniform"] = "uniform"
    oversampling: Optional[int] = None
    oversampling_seed: int = 1
    radiation: Literal["neutron", "xray"] = "xray"
    """

    ATTRIBUTES = [
        "intensity",
        "background",
        "back_absorption",
        "theta_offset",
        "sample_broadening",
        "name",
        "filename",
        "back_reflectivity",
        "R",
        "dR",
        "T",
        "dT",
        "L",
        "dL",
        "dQo",
        "resolution",
        "oversampling",
        "oversampling_seed",
        "radiation",
    ]
    if probe.filename is not None:
        s_probe = f'load4("{probe.filename}")'
    else:
        probe_dict = asdict(probe)
        s_probe = f"Probe(**{{{probe_dict}}})"
    return s_probe


def serialize_qprobe(probe: QProbe):
    """
    Serialize a QProbe instance to a string representation.

    Parameters:
        probe (QProbe): The QProbe instance to serialize.

    Returns:
        str: A string representation of the QProbe instance.
    """
    if probe.filename is not None:
        s_probe = f'load4("{probe.filename}")'
    else:
        s_probe_args = [
            f"intensity={probe.intensity.value}",
            f"background={probe.background.value}",
            f"back_absorption={probe.back_absorption.value}",
            f"back_reflectivity={probe.back_reflectivity}",
            f"filename=None",
            f"Q={probe.Q.tolist() if probe.Q is not None else 'None'}",
            f"dQ={probe.dQ.tolist() if probe.dQ is not None else 'None'}",
            f"R={probe.R.tolist() if probe.R is not None else 'None'}",
            f"dR={probe.dR.tolist() if probe.dR is not None else 'None'}",
            f'resolution="{probe.resolution}"',
        ]
        # Create the separator outside the f-string to avoid backslash in f-string expression (Python 3.10 compatibility)
        separator = ",\n    "
        s_probe = f"QProbe(\n    {separator.join(s_probe_args)},\n)"
    return s_probe
