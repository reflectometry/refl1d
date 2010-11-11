"""
KV O'Donovan, JA Borchers, CF Majkrzak, O Hellwig and EE Fullerton (2002)
Pinpointing Chiral Structures with Front-Back Polarized Neutron Reflectometry
Physical Review Letters 88(6) 067201(4)
http://dx.doi.org/10.1103/PhysRevLett.88.067201
"""
from periodictable import Pt, Fe, Ni, Si, O, formula
from refl1d import *

#from refl1d import ncnrdat
#import periodictable

# Compute slits from dT given in the staj file
slits = 0.03
instrument = ncnrdata.XRay(dLoL=0.005, slits_at_Tlo=slits)
probe = instrument.load('e1085009.log')
probe.log10_to_linear()  # data was stored as log_10 (R) rather than R

# Values from staj
Pt_cap  = SLD(name='Pt', rho=86.431, irho=42.41/(2*1.54))
NiFe    = SLD(name="NiFe", rho=63.121, irho=8.24/(2*1.54))
FePt    = SLD(name="FePt", rho=93.842, irho=32.2/(2*1.54))
Pt_seed = SLD(name="seed", rho=110.404, irho=42.41/(2*1.54))
glass   = SLD(name="glass", rho=15.086, irho=1.55/(2*1.54))


sample = (glass(0,17.53/2.35)
          | Pt_seed(22.9417, 20.72/2.35)
          | FePt(146.576, 20.22/2.35)
          | NiFe(508.784, 29.93/2.35)
          | Pt_cap(31.8477, 25.18/2.35)
          | air)

# grower values
if len(sys.argv)>2:
    init = sys.argv[2]
    if init == "grower":
        # Materials
        fPt = formula("Pt")
        fNiFe = formula("Ni80Fe20",density=0.8*Ni.density+0.2*Fe.density)
        fFePt = formula("Fe55Pt45",density=0.55*Fe.density+0.45*Pt.density)
        fglass = formula("SiO2",density=2.2) 
        # Note: glass density is usually in [2.2, 2.5].  To get the fitted
        # SLD of 15 requires a density of 1.85, which seems very low. This
        # author does not know what kind of glass was used in the experiment,
        # so a variety were checked such as the ones below.
        #fglass = mix_by_weight('SiO2',75,'Na2O',15,'CaO',10,density=2.52)
        #fglass = mix_by_weight('SiO2',73,'B2O3',10,'Na2O',8,'K2O',8,'CaO',1,density=2.2)
        
        # Bulk SLD
        Pt_cap.rho.value, Pt_cap.irho.value = fPt.xray_sld(1.54)
        NiFe.rho.value, NiFe.irho.value = fNiFe.xray_sld(1.54)
        FePt.rho.value, FePt.irho.value = fFePt.xray_sld(1.54)
        Pt_seed.rho.value, Pt_seed.irho.value = fPt.xray_sld(1.54)
        fglass.rho.value,fglass.irho_value = fglass.xray_sld(1.54)
        
        # Expected thickness/interface
        sample[1].thickness.value = 15
        sample[2].thickness.value = 200
        sample[3].thickness.value = 500
        sample[4].thickness.value = 25
        for i,L in enumerate(sample[1:-2]):
            L.interface.value = 10
    elif init == "fitted":
        pass
    else:
        raise ValueError("unknown initializer '%s'"%init)        

# Fit parameters
#probe.theta_offset.dev(0.01)  # accurate to 0.01 degrees

search = "open"
if len(sys.argv) > 1:
    search = sys.argv[1]
if search == "open": # Open set
    for i,L in enumerate(sample[0:-1]):
        if i>0: L.thickness.range(0,1000)
        L.interface.range(0,50)
        L.material.rho.range(0,200)
        L.material.irho.range(0,200)
elif search == "jiggle": # jiggle
    for i,L in enumerate(sample[0:-1]):
        if i>0: L.thickness.pmp(0,10)
        L.interface.pmp(0,10)
        L.material.rho.pmp(10)
        L.material.irho.pmp(10)
elif search == "grower": # grower
    sample[1].thickness.range(15,35)
    sample[2].thickness.range(150,250)
    sample[3].thickness.range(300,700)
    sample[4].thickness.range(15,35)
    for i,L in enumerate(sample[1:-2]):
        L.interface.pmp(100)
        L.material.rho.pmp(10)
        L.material.irho.pmp(10)

elif search == "d2d3": # d2 X d3
    #sample[2].thickness.range(0,400)
    #sample[3].thickness.range(0,1000)
    sample[2].thickness.range(50,400)
    sample[3].thickness.range(50,1000)
    sample[2].thickness.value = 400
    sample[3].thickness.value = 1000

else:
    raise ValueError("unknown search '%s'"%search)

M = Experiment(probe=probe, sample=sample)

# Needed by dream fitter
problem = FitProblem(M)
problem.dream_opts = dict(chains=20,draws=1000,burn=3000)
problem.name = "Example"
problem.title = "xray"
problem.store = "T"+search
