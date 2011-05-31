from refl1d import *

Si = SLD(name="Si", rho=2.0737, irho=2.376e-5)
Cu = SLD(name="Cu", rho=6.5535, irho=8.925e-4)
Ta = SLD(name="Ta", rho=3.8300, irho=3.175e-3)
TaOx = SLD(name="TaOx", rho=1.6325, irho=3.175e-3)
NiFe = SLD(name="NiFe", rho=9.1200, irho=1.032e-3)
CoFe = SLD(name="CoFe", rho=4.3565, irho=7.986e-3) # 60:40
IrMn = SLD(name="IrMn", rho=-0.21646, irho=4.245e-2)

sample = (Si(0,2.13) | Ta(38.8,2) 
          | MagneticSlab(NiFe(25.0,5), rhoM=1.4638, thetaM=270)
          | MagneticSlab(CoFe(12.7,5), rhoM=3.7340, thetaM=270) 
          | Cu(28,2)
          | MagneticTwist(CoFe(30.2,5), rhoM=[4.5102,1.7860], thetaM=[270,85])
          | IrMn(4.74,1.7)
          | Cu(5.148,2) | Ta(55.4895,2) | TaOx(47.42,3.5) | air)

instrument = NCNR.NG1()
probe = instrument.load_magnetic("n101Gc1.reflA")

experiment = Experiment(probe=probe, sample=sample)
problem = FitProblem(experiment)