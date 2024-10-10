import numpy
from refl1d.models import *

# FIT USING REFL1D 0.6.12
# === Data files ===
# instrument template, load s1, s2, sample_width, and sample broadening
# sample_broadening = FWHM - 0.5*(s1+s2)/(d1-d2)
# for NG1, d1 = 1905 mm, d2 = 355.6 mm
# FWHM = 0.057 deg
instrument = NCNR.NG1(slits_at_Tlo=(1.5, 0.5), sample_width=10, sample_broadening=0.02)
probe = instrument.load_magnetic("pmf13_800mT.reflA", back_reflectivity=False)
probe.xs[1], probe.xs[2] = None, None
probe.oversample(n=20)

# === Materials ===
b_FePt = Parameter(4.925, name="nuclear_FePt")
b_trans = Parameter(4.53, name="nuclear_trans")
b_PtMn = Parameter(2.004, name="nuclear_PtMn")

m_FePt = Parameter(126.7, name="magnetic_FePt")
m_trans = Parameter(100, name="magnetic_trans")
m_PtMn = Parameter(9.057, name="magnetic_PtMn")

p_FePt = m_FePt * (2.853e-3)
p_trans = m_trans * (2.853e-3)
p_PtMn = m_PtMn * (2.853e-3)

FePt = SLD(name="FePt++", rho=4.925)
PtMn = SLD(name="PtMn++", rho=2.004)
MgO = SLD(name="MgO", rho=6.01)

# === Sample ===
PtMn_nuc = FreeInterface(below=PtMn, above=FePt, dz=[1] * 3, dp=[1] * 3, name="PtMn gradient")
# FePt_nuc=FreeInterface(below=FePt,above=air,
#                               dz=[1]*4,dp=[1]*4,name="FePt nuclear")
FePt_nuc = FreeLayer(
    below=FePt,
    above=air,
    # z=[0,0.1,0.5,0.5,0.9,1],
    # rho=[4.925,4.925*0.9,4.925/2,4.925/2,4.925*0.1,0],
    z=numpy.linspace(0, 1, 6),
    rho=numpy.linspace(4.925, 0, 6),
    name="FePt nuclear",
)
FePt_mag = FreeMagnetism(z=numpy.linspace(0, 1, 6), rhoM=numpy.linspace(0.1, 0, 6), name="FePt magnetic")

sample = MgO(0, 1) | PtMn(375, 0) | PtMn_nuc(50, 0) | FePt_nuc(270, 0, FePt_mag) | air

# === Fit parameters ===
for p in FePt_nuc.z[1:-1]:
    p.range(0, 1)
for p in FePt_nuc.rho[1:-1]:
    p.range(0, 7)
FePt_mag.z = FePt_nuc.z
if 0:  # Tie magnetic SLD to nuc
    scale = Parameter(0.1, "FePt mag")
    FePt_mag.rhoM = FePt_nuc.rho * scale
else:  # Magnetic independent of nuc
    for p in FePt_mag.rhoM:
        p.range(0, 0.2)

# === Problem definition ===
M = Experiment(sample=sample, probe=probe, dz=1, dA=1)

problem = FitProblem(M)
problem.name = "pmf"
