
# Create materials
Si = Material('Si')
SiO2 = Material('SiO2', bulk_density=2.2)
Au = Material('Au')
Cr = Material('Cr')

# Typical mu-metal chemistry by weight (from Magnetic Shield Corp)
#    Ni:80.2%  Fe:14.1%  Mo:4.85%  Mn:0.50%  Si:0.30%
#    Cr,Co,C,Al,S,P:<0.02% max
mumetal = Mixture.bymass(materials=[Material('Ni'), 0.8,
                                    Material('Mo'), 0.045,
                                    Material('Fe')],
                         name="mu-metal",
                         )

H2O = Material('H2O', bulk_density=1)
D2O = Material('D2O', bulk_density=1)
D2O_50 = Mixture.byvolume(materials=[H2O, 0.5, D2O])
choline = Material('C5H14NO')
PO4 = Material('PO4')
COOH = Material('COOH')
CH2 = Material('CH2')


bilayer = CompositionSpace(solvent=D2O_50, thickness=300)
bilayer.add(Part(material=COOH, profile=Gaussian(), fraction=0.5))
bilayer.add(Part(material=CH2, profile=Gaussian(), fraction=0.5))
bilayer.add(Part(material=PO4, profile=Gaussian(), fraction=0.5))
bilayer.add(Part(material=choline, profile=Gaussian(), fraction=0.5))
bilayer.add(Part(material=choline, profile=Gaussian(), fraction=0.5))
bilayer.add(Part(material=PO4, profile=Gaussian(), fraction=0.5))
bilayer.add(Part(material=CH2, profile=Gaussian(), fraction=0.5))
bilayer.add(Part(material=COOH, profile=Gaussian(), fraction=0.5))

sample = Sample()
sample.substrate(D2O_50)
sample.add(bilayer)
sample.add(Slab(Au, thickness=Par.pm(20,10)))
sample.add(Slab(mumetal, thickness=Par.pm(40,10)))
sample.add(Slab(SiO2, thickness=Par.pm(10,5)))
sample.surround(Si)
