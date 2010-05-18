import sys; sys.path.append('../..'); sys.path.append('../../../dream')
from math import pi
from refl1d import *
#Probe.view = 'log' # log, linear, fresnel, or Q**4

#from refl1d import ncnrdat
import periodictable

instrument = ncnrdata.NG1(Tlo=0.35, slits_at_Tlo=0.21)
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
# rocking curve FWHM is 0.015 degrees
# rocking curve integrated intensity is about 2500
# Peak position uncertainty is given by dX = w/sqrt(I)
# Appl. Phys. A 74 [Suppl.], S112 - S114 (2002) / Digital Object Identifier (DOI) 10.1007/s003390201392
probe.theta_offset.dev(0.015/2.35/50)  # FWHM rocking curve is 0.015 degrees

# Sample broadening is determined by the following
dT = instrument.calc_dT(T=0.15,slits=(0.5,0.25))
print "rocking curve FWHM",0.015,"dT",dT,"sample broadening",0.015-dT

substrate_slab.interface.range(0,10)
substrate.rho.pmp(1)

MgO_slab.interface.range(0,10)
MgO.rho.pmp(30)
MgO_slab.thickness.range(0,120)

MgHx1_slab.interface.range(0,400)
MgHx1.rho.range(MgH2_sld*1.1,Mg_sld*1.1)
MgHx1_slab.thickness.range(0,1200)

MgHx2_slab.interface.range(0,200)
MgHx2.rho.range(MgH2_sld*1.1,Mg_sld*1.1)
MgHx2_slab.thickness.range(0,1200)

Pd_slab.interface.range(0,200)
Pd.rho.pmp(10)
Pd_slab.thickness.pmp(30)

# Do the fit
M = Experiment(probe=probe, sample=sample)
if 0:

    
    view = 'best'
    if view == 'mode':
        substrate_slab.interface.value = 39.8
        MgO_slab.interface.value = 1.2
        MgO.rho.value = 6.861
        MgO_slab.thickness.value = 53.6
        MgHx1_slab.interface.value = 21.6
        MgHx1.rho.value = 1.409
        MgHx1_slab.thickness.value = 1294.3
        MgHx2_slab.interface.value = 50.7
        MgHx2.rho.value = 0.49
        MgHx2_slab.thickness.value = 69.2
        Pd_slab.interface.value = 20.1
        Pd.rho.value = 4.006
        Pd_slab.thickness.value = 573.4

    elif view == 'best':
        substrate_slab.interface.value = 39.8
        MgO_slab.interface.value = 1.2
        MgO.rho.value = 6.86
        MgO_slab.thickness.value = 53.7
        MgHx1_slab.interface.value = 21.1
        MgHx1.rho.value = 1.407
        MgHx1_slab.thickness.value = 1294.3
        MgHx2_slab.interface.value = 50.5
        MgHx2.rho.value = 0.49
        MgHx2_slab.thickness.value = 69
        Pd_slab.interface.value = 20.1
        Pd.rho.value = 4.005
        Pd_slab.thickness.value = 574
    
    elif view == 'mean':
        substrate_slab.interface.value = 38.9
        MgO_slab.interface.value = 1.03
        MgO.rho.value = 6.861
        MgO_slab.thickness.value = 53.6
        MgHx1_slab.interface.value = 21.6
        MgHx1.rho.value = 1.409
        MgHx1_slab.thickness.value = 1295.3
        MgHx2_slab.interface.value = 50.4        
        MgHx2.rho.value = 0.49
        MgHx2_slab.thickness.value = 68.3
        Pd_slab.interface.value = 20.1
        Pd.rho.value = 4.006
        Pd_slab.thickness.value = 573.4

    result = preview(models=M)

elif 0:
    import refl1d.fitter
    problem = refl1d.fitter._make_problem(models=M,weights=None)
    for i,p in enumerate(problem.parameters):
        print i+1,p.name
    if 1:
        from dream import *
        print "loading data"
        state = load_state('De2_VATR_3', skip=4000)
        print "loaded"
        state.labels = [
'probe theta offset',
'substrate interface',
'substrate rho',
'MgO interface',
'MgO rho',
'MgO thickness',
'MgHx1 interface',
'MgHx1 rho',
'MgHx1 thickness',
'MgHx2 interface',
'MgHx2 rho',
'MgHx2 thickness',
'Pd interface',
'Pd rho',
'Pd thickness',
]

elif 0:
    result = preview(models=M)
elif 0:
    result = fit(models=M, fitter=None)
    #result.resample(samples=100, restart=True, fitter=DEfit)
    result.mcmc(samples=int(1e6))
    result.save('De2_VATR_2')
    result.show()
    result.show_stats()
else:
    # Version 5: run longer with larger population and MgO interface at 40
    #    20N for 5*2000 generations
    # Version 6: 
    #    let theta offset vary according to rocking curve width
    #    let substrate rho vary by 1%
    #    limit MgHxi thickness to 1200
    #    5N for 4*2500 generations
    state = draw_samples(models=M, chains=10, generations=2500, cycles=4)
    state.save('De2_VATR_6')
    state.show(portion=1)
