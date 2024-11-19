from refl1d.names import *

M = load_mlayer("mlayer.staj")
M.probe.log10_to_linear()
problem = FitProblem(M)
