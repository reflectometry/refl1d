from refl1d.models import *

substrate = SLD("substrate", 4.0)
L1 = SLD("L1", 2.0)
L2 = SLD("L2", 4.0)
surface = SLD("air", 0.0)

# Can't add surface/substrate magnetism yet...
# sample = (
#    substrate(  0,0,magnetism=Magnetism(rhoM=0.0,thetaM=90    ))
#    | L1     (200,0,magnetism=Magnetism(rhoM=1.0,thetaM=0.0001))
#    | L2     (200,0,magnetism=Magnetism(rhoM=1.0,thetaM=90    ))
#    | surface(  0,0,magnetism=Magnetism(rhoM=0.0,thetaM=90    ))
#    )
sample = (
    substrate(0, 0)
    | L1(200, 0, magnetism=Magnetism(rhoM=1.0, thetaM=0.0001))
    | L2(200, 0, magnetism=Magnetism(rhoM=1.0, thetaM=90))
    | surface(0, 0)
)

T = numpy.linspace(0, 3, 400)

xs = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
probeH = PolarizedNeutronProbe([xs] * 4, H=0.4, Aguide=270, name="H=0.4 T")
probe0 = PolarizedNeutronProbe([xs] * 4, H=0, Aguide=270, name="H=0.0 T")

M0 = Experiment(probe=probe0, sample=sample)
MH = Experiment(probe=probeH, sample=sample)

# M.simulate_data(5)

problem = FitProblem([M0, MH])
