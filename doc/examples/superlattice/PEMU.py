'''
Reference: Jomaa, H., Schlenoff, Macromolecules, 38 (2005), 8473-8480

Inter-diffusion properties of multilayer systems are of great interest in both hard and soft materials. Jomaa, et. al  has showed that reflectometry can be used to elucidate the kinetics of a diffusion process in polyelectrolytes multi-layers. Although the purpose of this paper was not to fit the presented system, it offers a good model for an experimentally relevant system for which information from neutron reflectometry can be obtained. In this model system we will show that we can create a model for this type of system and determine the relevant parameters through our optimisation scheme. 
'''

from refl1d.names import *

# Note: Parts of this problem would be better modelled with volume fractions (I think). I don't have documentation on the syntax for this though and so I use a standard model.

#These are materials for the substrate. They will not really need to be fitted because we know what they are from other techniques.(i.e.. We measured reflectometery on the system beforehand and know the characteristics.
chrome = Material('Cr')
gold = Material('Au')

#Deuterated reference layer. This is a labelled layer in the super lattice which will allow use to determine whether diffusion of PSS into PDADMA is occurring.
PDADMA_dPSS = SLD(name ='PDADMA dPSS',rho = 2.77)

#The rest of the layers will be undeuterated which means contrast between the bilayers will be minimal. We can refer to this as one layer.
PDADMA_PSS = SLD(name = 'PDADMA PSS',rho = 1.15)

#Because this is a bilayer system, it is convenient to use the repeat syntax.
bilayer = PDADMA_PSS(178,10) | PDADMA_dPSS(44.3,10)

#This is the sample. because we are expecting the top layer to act differently than the multilayer, I have added an extra piece on top. This probably isn't needed but I had trouble with the model building syntax.
sample = silicon(0,5) | chrome(30,3) | gold(120,5) | (PDADMA_PSS(178,10) | PDADMA_dPSS(44.3,10))*4 | PDADMA_PSS(178,10) | PDADMA_dPSS(44.3,10) | air

#I need to fit the roughness to determine how much diffusion has occurred. I have set the maximum interfacial roughness to be the thickness of the thin layer to simulate total diffusion. How will the software proceed? 
sample[3].interface.range(5,44.3)

#Along with roughness, the SLD of the deuterated section will fall. Here I allow the low SLD to move up and the high SLD to move down. We actually know these parameters better because we know that the high SLD is going probably not going to get to 11.5 and visa versa. 
bilayer[0].material.rho.range(11.5,27.7)
bilayer[1].material.rho.range(11.5,27.7)

#We know because of surface diffusion effects that the top layer will act differently than the other multilayers. This is fit separately to allow it freedom to deviate from the bilayer. Is this required or can we do this with the bilayer system?
sample[4].material.rho.range(11.5,27.7)
sample[4].interface.range(5,50)
sample[5].material.rho.range(20,27.7)


T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)


M = Experiment(probe=probe, sample=sample)
M.simulate_data(5)

problem = FitProblem(M)
