from refl1d.names import *
from refl1d.probe.data_loaders.polref_legacy_data_loader import load_probe_polref

dQoQ = 0.01
theta = 0.25
probe = load_probe_polref(filename="Ni58",
                          angle=0.25, dQoQ=0.01, name="Ni58",
                          pol_mode="pnr",
                          intensity=1.0, background=1e-7, back_reflectivity=False)

probe.pp.intensity.range(1e-1, 10)
probe.pp.background.range(1e-9, 1e-3)
probe.pp.sample_broadening.range(-(dQoQ*theta), 0.03)

# Set materials/SLDs
Si = Material(formula="Si")
Ni = Material(formula="Ni[58]")


Si_sub = Slab(material=Si, thickness=0, interface=5)
Ni_layer = Slab(material=Ni, thickness=1200, interface=5)

# Sample construction/Stack

sample = (Si_sub
          | Ni_layer(magnetism=Magnetism(rhoM=2.0, interface_above=5, interface_below=5, name="Ni Layer Sample 1"))
          | air
          )


# Fit params

Ni.density.pmp(-50, 0)

sample[Ni].magnetism.rhoM.range(0, 5)
sample[Ni].magnetism.dead_above.range(0, 100)
sample[Ni].magnetism.dead_below.range(0, 100)
sample[Ni].magnetism.interface_above.range(0, 50)
sample[Ni].magnetism.interface_below.range(0, 50)

sample[Ni].magnetism.rhoM.tags = ["magnetism", "sample"]
sample[Ni].magnetism.dead_above.tags = ["magnetism", "sample"]
sample[Ni].magnetism.dead_below.tags = ["magnetism", "sample"]
sample[Ni].magnetism.interface_above.tags = ["magnetism", "sample"]
sample[Ni].magnetism.interface_below.tags = ["magnetism", "sample"]


Ni_layer.thickness.range(0, 1500)
Ni_layer.thickness.tags = ["structure", "sample"]

Ni_layer.interface.range(0, 50)
Ni_layer.interface.tags = ["structure", "sample"]

Si_sub.interface.range(0, 50)
Si_sub.interface.tags = ["structure", "sample"]

zed = 2
step = False

experiment = Experiment(probe=probe, sample=sample, dz=zed, step_interfaces=step, auto_tag=True)

problem = FitProblem(experiment)


