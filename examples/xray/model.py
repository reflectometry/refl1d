"""
Single wavelength X-ray measurement.

The sample used in this data set is a magnetic thin film on glass,
measured with Cu K-alpha X-rays to obtain the structural parameters.

-----
    glass | FePt seed | 55:45 FePt | 80:20 NiFe | Pt cap
          |   2 nm    |   15 nm    |   50 nm    |  3 nm
-----

For a complete description of the system see [#ODonovan02]_

This model allows you to explore various initial conditions and search
ranges.

Usage
-----

Show the initial model as described by the sample grower::

    $ refl1d model.py grower --preview

Show the model fitted with reflpak::

    $ refl1d model.py --preview

Tune the reflpak fitted model::

    $ refl1d model.py jiggle

Fit the depth of neighbouring layers.  This is a 2-D problem with many
local minima::

    $ refl1d model.py d2Xd3

Perform a complete analysis on the system with very loose constraints::

    $ refl1d model.py open --fit=dream

References
----------

.. [#ODonovan02]_:
    KV O'Donovan, JA Borchers, CF Majkrzak, O Hellwig and EE Fullerton (2002).
    "Pinpointing Chiral Structures with Front-Back Polarized Neutron
    Reflectometry".  Physical Review Letters 88(6) 067201(4)
    http://dx.doi.org/10.1103/PhysRevLett.88.067201

"""
import sys
from periodictable import formula, xray_sld, Cu, Ni, Fe, Pt
from refl1d.names import *

# Determine what search we are going to perform
search = "rltest"
if len(sys.argv) > 1:
    search = sys.argv[1]

# Slits computed from dT given in the staj file
slits = 0.03
instrument = NCNR.XRay(dLoL=0.005, slits_at_Tlo=slits)
probe = instrument.load('e1085009.log')
probe.log10_to_linear()  # data was stored as log_10 (R) rather than R

# Values from staj file
glass = SLD(name="glass", rho= 15.086, irho= 0.503)
seed  = SLD(name="seed",  rho=110.404, irho=13.769)
FePt  = SLD(name="FePt",  rho= 93.842, irho=10.455)
NiFe  = SLD(name="NiFe",  rho= 63.121, irho= 2.675)
cap   = SLD(name='cap',   rho= 86.431, irho=13.769)

# Sample stack
sample = (glass(0,7.460)
          | seed(22.9417, 8.817)
          | FePt(146.576, 8.604)
          | NiFe(508.784, 12.736)
          | cap(31.8477, 10.715)
          | air)

# Layers from the stack
Lglass,Lseed,LFePt,LNiFe,Lcap,Lair = sample

# grower values
if search == "grower":
    # Materials
    fPt = formula("Pt")
    fNiFe = formula("Ni80Fe20",density=0.8*Ni.density+0.2*Fe.density)
    fFePt = formula("Fe55Pt45",density=0.55*Fe.density+0.45*Pt.density)
    # Note: we have been given the Cu K-alpha SLD of the glass substrate
    # but the formula itself was not provided.  The common glass
    # preparations below do not come close to matching the given SLD
    # unless density is lowered to about 1.85.  Since most glass has
    # a density in [2.2, 2.5], none of these formulas can be used, and we
    # will restrict ourselves to an approximate SLD provided by the grower.
    #fglass = formula("SiO2",density=2.2)
    #fglass = mix_by_weight('SiO2',75,'Na2O',15,'CaO',10,density=2.52)
    #fglass = mix_by_weight('SiO2',73,'B2O3',10,'Na2O',8,'K2O',8,'CaO',1,density=2.2)
    glass_sld = 15,0.5

    # Bulk SLD
    cap.rho.value, cap.irho.value = xray_sld(fPt, wavelength=Cu.Kalpha)
    NiFe.rho.value, NiFe.irho.value = xray_sld(fNiFe, wavelength=Cu.Kalpha)
    FePt.rho.value, FePt.irho.value = xray_sld(fFePt, wavelength=Cu.Kalpha)
    seed.rho.value, seed.irho.value = xray_sld(fFePt, wavelength=Cu.Kalpha)
    glass.rho.value,glass.irho_value = glass_sld

    # Expected thickness/interface
    Lseed.thickness.value =  15
    LFePt.thickness.value = 200
    LNiFe.thickness.value = 500
    Lcap.thickness.value =   25
    for i,L in enumerate(sample[1:-2]):
        L.interface.value = 10

# Fit parameters
#probe.theta_offset.dev(0.01)  # accurate to 0.01 degrees

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
    Lseed.thickness.range(10,50)
    LFePt.thickness.range(50,250)
    LNiFe.thickness.range(300,700)
    Lcap.thickness.range(10,50)
    for i,L in enumerate(sample[:-2]):
        L.interface.range(0,20)
        L.material.rho.pmp(20)
        L.material.irho.pmp(20)
elif search == "rltest": # grower
    Lseed.thickness.pmp(100)
    LFePt.thickness.pmp(100)
    LNiFe.thickness.pmp(100)
    Lcap.thickness.pmp(100)
    for i,L in enumerate(sample[:-2]):
        L.interface.pmp(30)
        L.material.rho.pmp(20)
        L.material.irho.pmp(20)

elif search == "d2Xd3": # d2 X d3
    #sample[2].thickness.range(0,400)
    #sample[3].thickness.range(0,1000)
    LFePt.thickness.range(50,400)
    LNiFe.thickness.range(50,1000)
    LFePt.thickness.value = 400
    LNiFe.thickness.value = 1000

else:
    raise ValueError("unknown search '%s'"%search)

M = Experiment(probe=probe, sample=sample)

# Needed by dream fitter
problem = FitProblem(M)
problem.dream_opts = dict(chains=20,draws=1000,burn=3000)
problem.name = "Example"
problem.title = "xray"
problem.store = "T"+search
