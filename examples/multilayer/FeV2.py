
import pylab
from mystic.param import Parameter as Par
from refl1d import Material, Mixture, Slab, Sample, Stack, Erf
from refl1d import ncnrdata

# === Data ===
instrument = ncnrdata.ANDR(Tlo=0.35, slits_at_Tlo=0.21)
probe = instrument.load("FeV.data", sample_broadening=0.0)
#probe.plot(); pylab.show()

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
P = []
MgO = Material('MgO', bulk_density=3.58)
V = Material('V')
FeV = Mixture.bymass(materials=[Material('Fe'), Material('V')],
                     weights=[0.5],
                     stretch=Par.pmp(1,1, name="FeV stretch"))
Pd = Material('Pd', stretch=Par.pmp(1,1, name="Pd stretch"))

class MaterialAdaptor(Material):
    def __init__(self, *args, **kw):
        Material.__init__(self, *args, **kw)


material_parameters = [
                       wrap(MgO,'stretch').pmp(1,1),
                       wrap(V,'stretch').pmp(1,1),
                       wrap(FeV,'stretch').pmp(1,1),

                       ]


# === Layers ===
# V:FeV interface is an error function somewhere in [0-5] A
Vrough = Erf(Par(0,5, name="V:FeV roughness"))

# MgO substrate
sample = Sample()
sample.substrate(MgO, interface=Erf(Par(0,5, name="MgO:V roughness")))

# 14 repeats of /V 15-45/FeV 5-25/
r1 = Stack(repeat=14)
r1.add(Slab(V, thickness=Par.pm(30,15,name="V1 thickness")),
       interface=Vrough)
r1.add(Slab(FeV, thickness=Par.pm(15,10,name="FeV1 thickness")),
       interface=Vrough)
sample.add(r1, interface=Vrough)

# 86 repeats of /V 15-45/FeV 5-25/
r2 = Stack(repeat=86)
r2.add(Slab(V, thickness=Par.pm(30,15,name="V2 thickness")),
       interface=Vrough)
r2.add(Slab(FeV, thickness=Par.pm(15,10,name="FeV2 thickness")),
       interface=Vrough)
sample.add(r2, interface=Vrough)

# /V 15-45/Pd 0-100/ cap, with independent V:Pd and Pd:Air roughness
sample.add(Slab(V, thickness=Par.pm(30,15, name="V cap thickness")),
           interface=Erf(Par(0,50, name="V:Pd roughness")))
sample.add(Slab(Pd, thickness=Par(0,200, name="Pd thickness")),
           interface=Erf(Par(0,50, name="Pd:air roughness")))


# === Fit ===
m1 = Experiment(sample=sample,probe=probe)
