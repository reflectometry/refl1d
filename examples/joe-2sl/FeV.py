
import numpy, pylab
from periodictable import elements
from mystic import Par
from reflectometry import Material, Alloy, Stack, Repeat

# === Materials ===

MgO = Material('MgO', density=Par.pmp(3.58,0.1))
V = Material(elements.V, packing=Par.pm(1,0.1))
FeV_alloy = Alloy.bymass(elements.Fe, Par.pm(0,100),
                         elements.V,
                         packing=Par.pm(1,0.1))
Pd = Material(elements.Pd, packing=Par.pm(1,0.1))

# === Layers ===

stack = Stack()
stack.base(MgO, roughness=Par(0,5))
r1 = Repeat(14)
r1.add(V, thickness=Par.pm(30,15), roughness=Vrough)
r1.add(FeV_alloy, thickness=Par.pm(10,5), roughness=Vrough)
stack.add(r1, roughness=Vrough)
Vrough = Par(0,3)
r2 = Repeat(86)
r2.add(V, thickness=Par.pm(30,15), roughness=Vrough)
r2.add(FeV_alloy, thickness=Par.pm(10,5))
stack.add(r2, roughness=Vrough)
stack.add(Pd, thickness=Par(0,100))
# no cap layer; use air
