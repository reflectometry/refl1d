
from refl1d import *

# Materials
MgO = Material('MgO', bulk_density=3.58)
V = Material('V')
FeV = Mixture('Fe',0.5,'V',fraction='mass')
Pd = Material('Pd')

# Sample description
V_FeV = V/14%3+FeV/35%5
sample = MgO%3 + 14*V_FeV + V/14%3 + FeV/60%5 + 85*V_FeV + V/14%2 + Pd

if 1: # density +/- 10%
    MgO.density.pmp(10)
    V.density.pmp(10)
    FeV.density.pmp(10)
    Pd.density.pmp(10)

if 1: # thickness +/- 10%
    for L in sample.layers():
        L.thickness.pmp(10)

if 1: # roughness in [0,15]
    for L in sample.layers():
        L.roughness.range(0,15)


# Show all parameters
print "All parameters"
for p in sample.parameters():
    print p.name, p.value, p.bounds

# Show Vanadium parameters
print "Vanadium parameters"
for p in sample.find('V .*'):
    print p.name, p.value, p.bounds

# Show all thicknesses
print "Thicknesses"
for p in sample.find(".* thickness"):
    print p.name, p.value, p.bounds

# === Sample ===
# Note on fittable parameters:
#   Par(xlo,xhi) is in [xlo, xhi]
#   Par.pm(x,dx) is in [x-dx, x+dx], with 2 digits precision
#   Par.pmp(x,p) is in [x-p*x/100, x+p*x/100], with 2 digits precision
#   The model itself may impose additional limits, such as x>0
# Note on material properties:
#   We can fit SLD directly as in traditional reflectivity software (using
#   SLD(rho,mu) as the material), but this does not give us the full power
#   of simultaneous X-ray and neutron refinement.  Instead we need to fit
#   the material density and compute the SLD from the composition.  Density
#   itself may be computed, either from the bulk density and a stretch
#   factor, or from a packing fraction and atomic radii, or from a unit
#   cell volume.  The example below uses stretch factors.  Bulk element
#   densities come from the periodic table.

# === Data ===
#pylab.plot(probe.Q, probe.dQ); pylab.show(); sys.exit()
#probe.plot(); pylab.show(); sys.exit()
instrument = ncnrdata.ANDR(Tlo=0.35, slits_at_Tlo=0.21)
probe = instrument.load("FeV.data", sample_broadening=0.0)
# === Fit ===
m1 = Experiment(sample=sample,probe=probe)

z,rho,mu = m1.step_profile()
pylab.plot(z,rho,'-',z,mu,'-')
pylab.legend(['rho','mu'])
pylab.show()
