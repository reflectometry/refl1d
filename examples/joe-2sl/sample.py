import fit
from offspec import *

class Shape:
    def __init__(self):
        self.constraints = []
    def on_top_of(self, other):
        self.center.z = 0.5*self.dim.z + 0.5*other.dim.z + other.center.z
    def no_overlap_in_plane(self, other):
        self.constraints.extend([
            (self.center.x-self.dim.x/2 > other.center.x+other.dim.x/2)
              | (self.center.x+self.dim.x/2 < other.center.x-other.dim.x/2),
            (self.center.y-self.dim.y/2 > other.center.x+other.dim.y/2)
              | (self.center.x+self.dim.y/2 < other.center.x-other.dim.y/2),
           ])

class Parallelepiped(Shape):
    def __init__(self,
            SLD=None,
            dim=[None,None,None],
            center = [None,None,None]):
        Shape.__init__(self)
        self.SLD = Par(SLD)
        self.dim.x = Par(dim[0])
        self.dim.y = Par(dim[1])
        self.dim.z = Par(dim[2])
        cx,cy,cz = center
        if cx == None: cx = 0.5*self.dim.x
        if cy == None: cy = 0.5*self.dim.y
        if cz == None: cz = 0.5*self.dim.z
        self.center.x = cx
        self.center.y = cy
        self.center.z = cz
    def parameters(self):
        return [self.dim.x,self.dim.y,self.dim.z, cx,cy,cz, self.SLD]
class Sphere(Shape):
    def __init__(self, SLD=None, radius=None, center=[None,None,None]):
        self.SLD = Par(SLD)
        self.radius = Par(radius)
        cx,cy,cz = center
        if cx == None: cx = self.radius
        if cy == None: cy = self.radius
        if cz == None: cz = self.radius
        self.dim.x = self.dim.y = self.dim.z = 2*self.radius
    def parameters(self):
        return [self.radius, cx,cy,cz, self.SLD]

class RotatedRod(Shape):
    def __init__(self, SLD=None, radius=None, length=None, angle=None,
                center=[None,None,None]):
        self.radius = Par(radius)
        self.length = Par(length)
        self.angle = Par(angle)
        self.dim.x = parameter.cos(Par(angle))

class Scene:
    def __init__(self, shapes):
        self.shapes = shapes

    def parameters(self):
        P = []
        for s in self.shapes:
            P.extend(self.shape.parameters)
        return P

A = Parallelepiped (SLD=2.7e-5, dim = [pm(5,2), pm(5,2), pm(5,2)])
B = Parallelepiped (SLD=pmp(2.7e-6,10),
                    dim = [pm(5,2), pm(5,2), A.dim.z],
                    )
C  = Layer(SLD=3.7e-5, thickness = 10)

# Option 1
C.is_base()
A.on_top_of(C)
B.on_top_of(C)
A.no_overlap_in_plane(B)
scene = Scene([A,B,C])

# Option 2
scene = Scene()
scene.is_base(C)
scene.above(A,C)
scene.above(B,C)
scene.no_overlap_in_plane(A,B)

class Unit_Cell:
    def __init__(self, Dxy = [None,None], scene = None):
        self.scene = scene
        Dx,Dy,Dz = dim
        for s in scene.shapes:
            self._constraints.extend([
            s.center.x+s.dim.x < self.Dx,
            s.center.y+s.dim.y < self.Dy,
            s.center.x-s.dim.x >= 0,
            s.center.y-s.dim.y >= 0,
            ])

        Dz =
        self.Dxyz = [Par(p) for p in dim]
    def parameters(self):
        L = self.scene.parameters()
        L.extend([self.Dx,self.Dy])
        return L
    def constraints(self):
        L = self.scene.constraints()
        L.extend(self._constraints)
        return L

class Calculator:
    def parameters(self):
        return self.lattice.parameters+self.beam.parameters+self.q_space.parameters + self.feature.parameters

    def


feature = Unit_Cell(Dxyz=[10,10,None],n=[50,50,50],scene = scene)
q_space = Q_space(qlo=[-0.001,0.0,0.002], qhi=[0.001,0.001,0.06],
                  shape=[200,100,300])
lattice = Lattice(repeats=[20,20,1])
beam = Beam (wavelength=5.0,resolution=0.05)
sample = Calculator(lattice,beam,q_space,feature)
result = fit.DE(problem = sample)
