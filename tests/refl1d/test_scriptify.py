import os
from pathlib import Path
import shutil

import pytest
from bumps.cli import load_model
from refl1d.webview.server.scriptify import serialize_fitproblem


@pytest.fixture
def temp_model_dir(tmp_path):
    """
    Create a temporary directory for scriptify output with the required data file.

    This fixture:
    - Creates a temporary directory using pytest's tmp_path
    - Copies the e1085009.log data file into it
    - Yields the path to the temporary directory
    - Automatically cleans up after the test

    Returns
    -------
    Path
        Path to the temporary directory containing the data file
    """
    # Get the path to the original data file
    example_dir = Path(__file__).parent.parent.parent / "doc" / "examples" / "xray"
    data_file = example_dir / "e1085009.log"

    # Copy the data file to the temporary directory
    shutil.copy(data_file, tmp_path / "e1085009.log")

    # Yield the temporary directory path
    yield tmp_path

    # Cleanup is automatic with tmp_path


def test_scriptify_refl1d_model(temp_model_dir):
    """Test that a refl1d model can be scriptified and re-loaded."""
    # Load the example model
    model_path = Path(__file__).parent.parent.parent / "doc" / "examples" / "xray" / "model.py"
    fit_problem = load_model(str(model_path))

    # Scriptify the fit problem
    script = serialize_fitproblem(fit_problem)

    assert script == OUTPUT_SCRIPT

    # Write the script to the temporary directory
    temp_script_path = temp_model_dir / "reloaded_model.py"
    temp_script_path.write_text(script)

    # Load the model from the scriptified version (it will find e1085009.log in temp_model_dir)
    # Need to change to temp directory so relative path works
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(temp_model_dir)
        reloaded_fit_problem = load_model(str(temp_script_path))

        # Compare key attributes of the original and reloaded fit problems
        assert len(list(fit_problem.models)) == len(list(reloaded_fit_problem.models))
        for original_model, reloaded_model in zip(fit_problem.models, reloaded_fit_problem.models):
            assert len(original_model.sample.layers) == len(reloaded_model.sample.layers)
            for orig_layer, reload_layer in zip(original_model.sample.layers, reloaded_model.sample.layers):
                assert orig_layer.thickness.value == reload_layer.thickness.value
                assert orig_layer.interface.value == reload_layer.interface.value
                if hasattr(orig_layer.material, "rho"):
                    assert orig_layer.material.rho.value == reload_layer.material.rho.value
                if hasattr(orig_layer.material, "irho"):
                    assert orig_layer.material.irho.value == reload_layer.material.irho.value
    finally:
        os.chdir(original_cwd)


OUTPUT_SCRIPT = """\
from refl1d.names import *

################
# Experiment_1 #
################


probe_1 = load4("e1085009.log")

# probe_1.intensity.range(0.8, 1.2) # intensity e1085009
# probe_1.background.range(0.0, 0.0001) # background e1085009
# probe_1.theta_offset.range(-inf, inf) # theta_offset e1085009
# probe_1.sample_broadening.range(-inf, inf) # sample_broadening e1085009

slabs_1 = [
    Slab(name="glass", thickness=0.0, interface=7.46, material=SLD(name="glass", rho=15.086, irho=0.503), magnetism=None),
    Slab(name="seed", thickness=22.9417, interface=8.817, material=SLD(name="seed", rho=110.404, irho=13.769), magnetism=None),
    Slab(name="FePt", thickness=146.576, interface=8.604, material=SLD(name="FePt", rho=93.842, irho=10.455), magnetism=None),
    Slab(name="NiFe", thickness=508.784, interface=12.736, material=SLD(name="NiFe", rho=63.121, irho=2.675), magnetism=None),
    Slab(name="cap", thickness=31.8477, interface=10.715, material=SLD(name="cap", rho=86.431, irho=13.769), magnetism=None),
    Slab(name="Vacuum", thickness=0.0, interface=0.0, material=air, magnetism=None),
]

sample_1 = Stack(slabs_1)

#####################
# Sample Parameters #
#####################

# slabs_1[0].thickness.range(0.0, inf) # glass thickness
slabs_1[1].thickness.range(0, 46) # seed thickness
slabs_1[2].thickness.range(0, 300) # FePt thickness
slabs_1[3].thickness.range(0, 1100) # NiFe thickness
slabs_1[4].thickness.range(0, 64) # cap thickness
# slabs_1[5].thickness.range(0.0, inf) # Vacuum thickness

slabs_1[0].interface.range(0, 15) # glass interface
slabs_1[1].interface.range(0, 18) # seed interface
slabs_1[2].interface.range(0, 18) # FePt interface
slabs_1[3].interface.range(0, 26) # NiFe interface
# slabs_1[4].interface.range(0.0, inf) # cap interface
# slabs_1[5].interface.range(0.0, inf) # Vacuum interface

slabs_1[0].material.rho.range(12.0, 18.2) # glass rho
slabs_1[1].material.rho.range(88, 133) # seed rho
slabs_1[2].material.rho.range(75, 113) # FePt rho
slabs_1[3].material.rho.range(50, 76) # NiFe rho
# slabs_1[4].material.rho.range(-inf, inf) # cap rho

slabs_1[0].material.irho.range(0.4, 0.61) # glass irho
slabs_1[1].material.irho.range(11.0, 16.6) # seed irho
slabs_1[2].material.irho.range(8.3, 12.600000000000001) # FePt irho
slabs_1[3].material.irho.range(2.1, 3.3000000000000003) # NiFe irho
# slabs_1[4].material.irho.range(-inf, inf) # cap irho



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

experiment_1 = Experiment(sample=sample_1, probe=probe_1, dz=1.0, step_interfaces=False, dA=1.0)

##############
# FitProblem #
##############

problem = FitProblem([experiment_1])\
"""

if __name__ == "__main__":
    test_scriptify_refl1d_model()
