from pathlib import Path

import numpy as np

from bumps.names import FitProblem, Parameter
from refl1d.names import Experiment, NeutronProbe, load_orso_sample


sample_file = Path(__file__).parent / "orso_substacks_sample.yml"
sample = load_orso_sample(sample_file)

# Share a fittable CH2 thickness across the repeats
CH2_thickness = Parameter(value=0.0, name="CH2_thickness")
for layer in sample.layers:
    if layer.name == "CH2":
        CH2_thickness.value = float(layer.thickness)
        layer.thickness = CH2_thickness
CH2_thickness.pmp(20)

# mixture = refl1d_sample.layers[-1].material
probe = NeutronProbe(T=np.linspace(0.0, 5.0, 150), L=[5.0])
M = Experiment(probe=probe, sample=sample)

# Verify that the loaded sample produces a reflectivity without crashing.
M.reflectivity()
if 0:  # Show hot execution time after numba compilation; 1.2 us on apple M2
    import time

    t0 = time.perf_counter_ns()
    M.reflectivity()
    dt = time.perf_counter_ns() - t0
    print(f"run time: {dt * 1e-3:.2f} μs")

problem = FitProblem([M])
