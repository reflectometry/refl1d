import pytest
import numpy as np
from refl1d.names import Material, Slab, Magnetism, Experiment, FitProblem, air
from refl1d.probe.data_loaders.polref_legacy_data_loader import load_probe_polref

def test_ni58_model_loading():
    # Parameters from Ni58_model.py
    dQoQ = 0.01
    theta = 0.25

    # The files Ni58_d.dat and Ni58_u.dat are in the same folder as this test
    # We need to pass the correct path to load_probe_polref
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))

    probe = load_probe_polref(filename="Ni58",
                              angle=theta, dQoQ=dQoQ, name="Ni58",
                              pol_mode="pnr",
                              intensity=1.0, background=1e-7, back_reflectivity=False,
                              path=current_path)

    # Verify probe creation
    from refl1d.names import PolarizedNeutronProbe
    assert isinstance(probe, PolarizedNeutronProbe)
    assert probe.mm is not None
    assert probe.pp is not None
    assert probe.mp is None
    assert probe.pm is None

    # Verify data was actually loaded (Ni58_d.dat and Ni58_u.dat exist)
    # We check if the Q array is non-empty
    assert len(probe.mm.Q) > 0
    assert len(probe.pp.Q) > 0

    # Setup model as in Ni58_model.py
    Si = Material(formula="Si")
    Ni = Material(formula="Ni[58]")

    Si_sub = Slab(material=Si, thickness=0, interface=5)
    Ni_layer = Slab(material=Ni, thickness=1200, interface=5)

    sample = (Si_sub
              | Ni_layer(magnetism=Magnetism(rhoM=2.0, interface_above=5, interface_below=5, name="Ni Layer Sample 1"))
              | air
              )

    # Fit params
    Ni.density.pmp(-50, 0)
    sample[Ni].magnetism.rhoM.range(0, 5)
    sample[Ni].magnetism.dead_above.range(0, 10) # Adjusted range for test stability if needed, but keeping original
    sample[Ni].magnetism.dead_below.range(0, 10)
    sample[Ni].magnetism.interface_above.range(0, 50)
    sample[Ni].magnetism.interface_below.range(0, 50)

    Ni_layer.thickness.range(0, 1500)
    Ni_layer.interface.range(0, 50)
    Si_sub.interface.range(0, 50)

    zed = 2
    step = False

    # Verify Experiment and FitProblem creation
    experiment = Experiment(probe=probe, sample=sample, dz=zed, step_interfaces=step, auto_tag=True)
    problem = FitProblem(experiment)

    assert experiment is not None
    assert problem is not None

