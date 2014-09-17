# Generate a completely random film on Si to test fitting.
#
# For example, the following generates a random film with three layers:  
#  
#   refl1d model.py 3 --simrandom --seed=101 --preview
#
# A different model will be generated for each different seed.  If 
# no seed is given, then the random number generator will be seeded 
# with noise.
#
# To test the fitting engine, you will want to use --shake to set
# a random initial value before starting the fit:
#
#   refl1d model.py 3 --simrandom --seed=101 --shake --fit=amoeba
#
# You will find that the amoeba fitter does not work well for
# random models.  Dream performs a bit better, able to recover
# models of 1-2 layers.
#

from refl1d.names import *

n = int(sys.argv[1]) if len(sys.argv)>1 else 2
materials = [SLD("L%d"%i,rho=1) for i in range(1,n+1)]
layers = [L(100,5) for L in materials]
sample = silicon(0,5) | layers | air

sample[0].interface.range(0,200)
for L in layers:
    L.material.rho.range(-10,10)
    L.thickness.range(0,1000)
    L.interface.range(0,200)

T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = Experiment(probe=probe, sample=sample)
problem = FitProblem(M)
