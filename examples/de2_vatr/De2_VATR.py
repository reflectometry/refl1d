import sys; sys.path.append('../..')

import refl1d
from refl1d.stajconvert import load_mlayer

M = load_mlayer("De2_VATR.staj")
for i in range(1,len(M.sample)-1):
    M.sample[i].thickness.pmp(10)
    M.sample[i].interface.pmp(10)
    M.sample[i].material.rho.pmp(10)
    M.sample[i].material.irho.pmp(10)
#refl1d.preview(models=M)
result = refl1d.fit(models=M) #, fitter=None)
result.resample(samples=100)
result.save('De2_VATR')
result.show()
result.show_stats()

