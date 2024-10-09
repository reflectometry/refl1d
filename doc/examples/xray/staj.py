from refl1d.models import *

M = load_mlayer("mlayer.staj")
M.probe.log10_to_linear()
problem = FitProblem(M)
