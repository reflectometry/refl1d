import sys; sys.path.append('../..')

#TODO: xray has smaller beam spot
# => smaller roughness
# => thickness depends on where the beam spot hits the sample
# Xray thickness variance = neutron roughness - xray roughness

import pylab
from refl1d import *
Probe.view = 'log' # log, linear, fresnel, or Q**4

# Data files
instrument = ncnrdata.ANDR(Tlo=0.5, slits_at_Tlo=0.2, slits_below=0.1)
probe1 = instrument.load('n5hl1_Short.refl')
probe2 = instrument.load('n5hl2_Short.refl')

# Sample description

# Have two layers of hydrated Mg + layer of Pd on top of sapphire.
# In one measurement, the Pd is hydrated, but in the other it is not.
# When the Pd is hydrated, the layer expands to make room for the
# hydrogens, but otherwise retains the same number of Pd atoms.  With
# careful calculations we can model the expansion factor and the number
# of H per Pd.
#
# Here are the parameters we will use for the Pd layer in the two samples.
#    w_Pd       : thickness of the Pd layer in sample 1
#    r_Pd       : density of Pd layer in sample 1
#    Pd_stretch : stretch factor for Pd from insertion of H in sample 2
#    nH         : number of H per Pd in sample 2 (0,10)
#
# Derived quantities:
#    w_PdH = w_Pd*Pd_stretch                       : width of the <Pd,H>
#    r_Pd_in_PdH = r_Pd * w_Pd/w_PdH               : density of Pd in <Pd,H>
#                = r_Pd / Pd_stretch
#    V_cell = Pd.mass in grams / r_Pd_in_PdH       : unit cell volume in <Pd,H>
#    r_H_in_PdH = nH * H.mass in grams / V_cell    : density of H in <Pd,H>
#               = nH * H.mass/Pd.mass * r_Pd_in_PdH
#               = nH * H.mass/Pd.mass * r_Pd / Pd_stretch
#    r_PdH = r_Pd_in_PdH + r_H_in_PdH              : density of <Pd,H>
#          = r_Pd / Pd_stretch * (1 + nH * H.mass/Pd.mass)

# ====== Models =======
Mg = Material('Mg')
Pd = Material('Pd')
MgH2 = Material('MgH2',density=1.45)
MgO = Material('MgO',density=3.58)
MgO2 = Material('MgO2',density=3)
MgO2H2 = Material('Mg(OH)2', density=2.3446)
Mg_MgH2a = Mixture.byvolume(Mg,MgH2,30) # 70% Mg / 30% MgH2
Mg_MgH2b = Mixture.byvolume(Mg,MgH2,10) # 90% Mg / 10% MgH2

nH = Parameter(1, name='#H in <Pd+kH>')
Pd_stretch = Parameter(1.1, name='<Pd+kH> stretch')
PdH = Compound(('Pd',1,'H',nH), name='<Pd+kH>')
PdH.density = (Pd.density / Pd_stretch
               * (1 + nH * (elements.H.mass/elements.Pd.mass)))

sample1 = (sapphire%0.85 + Mg_MgH2a/480%31.6 + Mg_MgH2b/790.286%0.43 
           + Pd/555.35%0.43 + air)
sample2 = sample1[:]
sample2[3] = PdH%0.43
sample2[3].thickness = sample1[3].thickness * Pd_stretch

#TODO: load saved results as new starting point

# ===== Fit parameters ====

#sapphire.fitby('bulk_density')
#sapphire.bulk_density.pmp(5)
probe1.theta_offset.dev(0.01)
probe2.theta_offset.dev(0.01)
Pd.bulk_density.pmp(10)
Mg_MgH2a.fraction[0].range(0,100)
#Mg_MgH2b.fraction[0].range(0,100)
Pd_stretch.range(1,2)
nH.range(0,10)

# Thickness and roughness
for L in sample1[1:4]: L.thickness.pmp(100)
for L in sample1[0:4]: L.interface.range(0,5)
#sample1[3].thickness.range(0,200)
sample1[1].interface.range(0,100)


# ==== Run the fit ====
exp1 = Experiment(probe=probe1, sample=sample1)
exp2 = Experiment(probe=probe2, sample=sample2)
models = (exp1,exp2)

preview(models=models)
#fit(models=models, npop=10)
