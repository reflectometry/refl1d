
import pylab
from mystic.parameter import Parameter as Par
from refl1d import Material, Mixture, Slab, Sample, Stack, Erf, Experiment
from refl1d import ncnrdata

# === Data ===
instrument = ncnrdata.ANDR(Tlo=0.35, slits_at_Tlo=0.21)
probe = instrument.load("FeV.data", sample_broadening=0.0)
#pylab.plot(probe.Q, probe.dQ); pylab.show(); sys.exit()
#probe.plot(); pylab.show(); sys.exit()

# === Materials ===
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
MgO = Material('MgO', bulk_density=3.58, stretch=Par.pmp(1,1))
V = Material('V', stretch=Par.pmp(1,1), use_incoherent=False)
FeV = Mixture.bymass(materials=[Material('Fe'),
                                Material('V', use_incoherent=False)],
                     fractions=[Par.limits(0, 1, name="FeV iron portion")],
                     stretch=Par.pmp(1,1), name="FeV")
Pd = Material('Pd', stretch=Par.pmp(1,1, name="Pd stretch"))

# === Layers ===
# V:FeV interface is an error function somewhere in [0-5] A
Vrough = Erf(Par.limits(0,5, name="V:FeV roughness"))

# MgO substrate
sample = Sample()
sample.substrate(MgO, interface=Erf(Par.limits(0,5, name="MgO:V roughness")))

Vslab = Slab(V, thickness=Par.pm(30,15,name="V thickness"))
FeVslab = Slab(FeV, thickness=Par.pm(15,10,name="FeV thickness"))

# 14 repeats of /V 15-45/FeV 5-25/
r1 = Stack(repeat=14)
r1.add(Vslab, interface=Vrough)
r1.add(FeVslab, interface=Vrough)
sample.add(r1, interface=Vrough)

# 1 thick V/FeV
Vslab = Slab(V, thickness=Par(0,60,name="V slip thickness"))
FeVslab = Slab(FeV, thickness=Par(0,30,name="FeV slip thickness"))

# 85 repeats of /V 15-45/FeV 5-25/
r2 = Stack(repeat=86)
r2.add(Vslab, interface=Vrough)
r2.add(FeVslab, interface=Vrough)
sample.add(r2, interface=Vrough)

# /V 15-45/Pd 0-100/ cap, with independent V:Pd and Pd:Air roughness
sample.add(Slab(V, thickness=Par.pm(30,15, name="V cap thickness")),
           interface=Erf(Par(0,50, name="V:Pd roughness")))
sample.add(Slab(Pd, thickness=Par.limits(0,200, name="Pd thickness")),
           interface=Erf(Par.limits(0,50, name="Pd:air roughness")))


# === Fit ===
m1 = Experiment(sample=sample,probe=probe)

z,rho,mu = m1.step_profile()
pylab.plot(z,rho,'-',z,mu,'-')
pylab.legend(['rho','mu'])
pylab.show()
