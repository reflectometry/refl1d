from refl1d import preview
from refl1d.stajconvert import load_mlayer

M = load_mlayer("mlayer.staj")
M.probe.log10_to_linear()
M.probe.view = 'log'
preview(models=M)
