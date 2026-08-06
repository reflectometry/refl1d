import yaml
from pathlib import Path
from refl1d.probe import NeutronProbe
from orsopy.fileio.data_source import Sample
from refl1d.probe.data_loaders import orso

sample_file = Path(__file__).parent / "orso_substacks_sample.yml"
substacks = yaml.safe_load(open(sample_file, "r").read())

orso_sample = Sample(name="substacks example", model=substacks)
orso_model = orso_sample.model
refl1d_sample = orso.orso_samplemodel_converter(orso_sample.model)
mixture = refl1d_sample.layers[-1].material
probe = NeutronProbe(T=[0.01], L=[5.0])
