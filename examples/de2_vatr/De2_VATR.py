import sys; sys.path.append('../..')
from math import pi

from refl1d import *
#from refl1d import ncnrdat
import periodictable

instrument = ncnrdata.ANDR(Tlo=0.35, slits_at_Tlo=0.21)
probe = instrument.load('n6hd2002E.refl')

Mg_sld = periodictable.formula('Mg',density=1.74).neutron_sld(wavelength=instrument.wavelength)[0]
MgH2_sld = periodictable.formula('MgH2',density=1.45).neutron_sld(wavelength=instrument.wavelength)[0]

# Sample
substrate = SLD(name='substrate', rho=2.86125e-4/(16e-6*pi), irho=2.85859e-10/1e-6)
MgO = SLD(name='MgO', rho=2.95032e-4/(16e-6*pi), irho=1.324e-10/1e-6)
MgHx1 = SLD(name='MgHx1', rho=1.71058e-5/(16e-6*pi), irho=6.59e-10/1e-6)
MgHx2 = SLD(name='MgHx2', rho=8.701250e-5/(16e-6*pi), irho=6.59e-10/1e-6)
Pd = SLD(name='Pd', rho=2.02097e-4/(16e-6*pi), irho=1.30548e-8/1e-6)
vacuum = SLD(name='Vacuum',rho=0, irho=0)

substrate_slab = Slab(substrate, interface=1e-10/2.35)
MgO_slab = Slab(MgO,thickness=64.0154,interface=1e-10/2.35)
MgHx1_slab = Slab(MgHx1,thickness=316.991,interface=197.324/2.35)
MgHx2_slab = Slab(MgHx2,thickness=1052.77,interface=78.2023/2.35)
Pd_slab = Slab(Pd,thickness=567.547,interface=48.6865/2.35)

sample = (substrate_slab + MgO_slab + MgHx1_slab + MgHx2_slab + Pd_slab + vacuum)

# Fit parameters
MgO.rho.pmp(30)
MgHx1.rho.range(MgH2_sld*1.1,Mg_sld*1.1)
MgHx2.rho.range(MgH2_sld*1.1,Mg_sld*1.1)
Pd.rho.pmp(10)

MgO_slab.thickness.pmp(100)
MgHx1_slab.thickness.range(0,1500)
MgHx2_slab.thickness.range(0,1500)
Pd_slab.thickness.pmp(30)

substrate_slab.interface.range(0,20)
MgO_slab.interface.range(0,20)
MgHx1_slab.interface.range(0,400)
MgHx2_slab.interface.range(0,200)
Pd_slab.interface.range(0,200)

# Do athe fit
M = Experiment(probe=probe, sample=sample)
if 0:
    result = preview(models=M)
else:
    result = fit(models=M, fitter=None)
    #result.resample(samples=100, restart=True, fitter=DEfit)
    result.mcmc(samples=int(1e6))
    result.save('De2_VATR_2')
    result.show()
    result.show_stats()

