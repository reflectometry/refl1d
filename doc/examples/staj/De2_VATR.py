from refl1d.names import *

# Load neutron model and data from staj file
# Layer names are ordered from substrate to surface, and defaults to
# the names in the original staj file.
# Model name defaults to the data file name
layers = ["sappire", "MgO", "MgHx1", "MgHx2", "Pd", "air"]
M = load_mlayer("De2_VATR.staj", layers=layers, name="n6hd2")

# Set thickness/roughness fitting parameters to +/- 20 %
# Set SLD to +/- 5% for all but the incident medium and the substrate.
for L in M.sample[1:-1]:
    L.thickness.pmp(20)
    L.interface.pmp(20)
    L.material.rho.pmp(5)

# Let the substrate SLD vary by 2%
M.sample[0].material.rho.pmp(2)
M.sample[0].interface.range(0, 20)
M.sample[1].interface.range(0, 20)

problem = FitProblem(M)
problem.name = "Desorption 2"
