import sys; sys.path.append('../..')

import refl1d
from refl1d.stajconvert import load_mlayer

M = load_mlayer("De2_VATR.staj")
for i in range(1,len(M.sample)-1):
    M.sample[i].thickness.pmp(30)
    M.sample[i].interface.pmp(30)
    M.sample[i].material.rho.pmp(10)
    #M.sample[i].material.irho.pmp(10)
if 1:
    refl1d.preview(models=M)
else:
    result = refl1d.fit(models=M) #, fitter=None)
    result.resample(samples=100, fitter=refl1d.DEfit)
    result.save('De2_VATR_staj')
    result.show()
    result.show_stats()

