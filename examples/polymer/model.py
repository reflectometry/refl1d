import sys; sys.path.append("../..")
from copy import copy
from pylab import *
from periodictable import nsf
from refl1d import *

# Attached please find two data sets for a tethered  approximately 10 nm thick 
# deuterated polystyrene chains in deuterated and hydrogenated toluene. 
# 10 nm thickness is for dry conditions and I am assuming these chains will 
# swell to 14-18 nm thickness once they are in toluene.
#    10ndt is for deuterated toluene case
#    10nht is for hydrogenated toluene case
# I also have to tell you that these chains are bound to the substrate by 
# using an initiator layer between substrate and brush chains. So in your 
# model you should have a silicon layer, silicon oxide layer, initiator layer 
# which is mostly hydrocarbon and scattering length density should be between 
# 0 and 1.5 depending on how much solvent is in the layer. Then you have the 
# swollen brush chains and at the end bulk solvent. When we do these swelling 
# measurements beam penetrate the system from the silicon side and the bottom 
# layer is deuterated or hydrogenated toluene.



## Composition based approach.
#deutrated_density = formula("C6H5C2D3").mass/formula("C6H5C2H3").mass
#D_polystyrene = Material("C6H5C2D3", density=0.909*deuterated_density)
#SiOx = Material("SiO2", density=2.634)
#alkane = Material("C8H18",density=0.703)  # Octane formula and density
#deutrated_density = formula("C6H5CD3").mass/formula("C6H5CH3").mass
#H_toluene = Material("C6H5CH3", density=0.8669)
#D_toluene = Material("C6H5CD3", density=0.8669*deuterated_density)
#H_initiator = Compound.byvolume(alkane, H_toluene, 10)
#D_initiator = Compound.byvolume(alkane, D_toluene, H_initiator.fraction[0])

## Direct SLD approach
D_polystyrene = SLD(name="D-PS",rho=6.2)
SiOx = SLD(name="SiOx",rho=3.47)
H_toluene = SLD(name="H-toluene",rho=0.94)
D_toluene = SLD(name="D-toluene",rho=5.66)
H_initiator = SLD(name="H-toluene+initiator",rho=0)
D_initiator = SLD(name="D-toluene+initiator",rho=1.5)

D_polymer_layer = TetheredPolymer(polymer=D_polystyrene, solvent=D_toluene,
                                  phi=100, head=50, tail=200, Y=2)
H_polymer_layer = copy(D_polymer_layer)
H_polymer_layer.solvent = H_toluene

D = silicon%2 + SiOx/10%2 + D_initiator/5%2 + D_polymer_layer/500%0 + D_toluene
H = silicon + SiOx + H_initiator + H_polymer_layer + H_toluene
# Share thickness and interface
for i,_ in enumerate(D):
    H[i].thickness = D[i].thickness
    H[i].interface = D[i].interface

if 0:
    D.plot()
    suptitle('D-toluene')
    figure()
    H.plot()
    suptitle('H-toluene')
    show()
                                                  