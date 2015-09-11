import inspect

from numpy import real, imag, asarray

from bumps.parameter import Parameter
from refl1d.model import Layer
from refl1d.magnetism import BaseMagnetism
from refl1d import util

class FunctionalProfile(Layer):
    """
    Generic profile function

    Parameters:

        *thickness* the thickness of the layer
        *interface* the roughness of the surface [not implemented]
        *profile* the profile function, suitably parameterized
        *tol* is the tolerance for considering values equal
        *magnetism* magnetic profile associated with the layer
        *name* is the layer name

    The profile function takes a depth vector *z* returns a density vector
    *rho*. For absorbing profiles, return complex vector *rho + irho*1j*.


    Fitting parameters are the available named arguments to the function.
    The first argument is a depth vector, which is the array of depths at
    which the profile is to be evaluated.  It is guaranteed to be increasing,
    with step size 2*z[0].

    Initial values for the function parameters can be given using name=value.
    These values can be scalars or fitting parameters.  The function will
    be called with the current parameter values as arguments.  The layer
    thickness can be computed as :func:`layer_thickness`.

    There is no mechanism for querying the larger profile to determine the
    value of the *rho* at the layer boundaries.  If needed, this information
    will have to be communicated through shared parameters.  For example::

        L1 = SLD('L1', rho=2.07)
        L3 = SLD('L3', rho=4)
        def linear(z, rhoL, rhoR):
            rho = z * (rhoR-rhoL)/(z[-1]-z[0]) + rhoL
            return rho
        profile = FunctionalProfile(100, 0, profile=linear,
                                    rhoL=L1.rho, rhoR=L3.rho)
        sample = L1 | profile | L3
    """
    RESERVED = ('thickness','interface','profile','tol','magnetism','name')

    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(self, thickness=0, interface=0, profile=None, tol=1e-3,
                 magnetism=None, name=None, **kw):
        if not name: name = profile.__name__
        if interface != 0: raise NotImplementedError("interface not yet supported")
        if profile is None: raise TypeError("Need profile")
        self.name = name
        self.thickness = Parameter.default(thickness, name=name+" thickness")
        self.interface = Parameter.default(interface, name=name+" interface")
        self.profile = profile
        self.tol = tol
        self.magnetism = magnetism

        self._parameters = _set_vars(self, name, profile, kw, self.RESERVED)

    def parameters(self):
        P = dict( (k,getattr(self,k)) for k in self._parameters)
        P['thickness'] = self.thickness
        #P['interface'] = self.interface
        return P

    def render(self, probe, slabs):
        Pw,Pz = slabs.microslabs(self.thickness.value)
        if len(Pw)== 0: return
        #print kw
        phi = asarray(self.profile(Pz,**self._fpars()))
        if phi.shape != Pz.shape:
            raise TypeError("profile function '%s' did not return array phi(z)"
                            %self.profile.__name__)
        Pw,phi = util.merge_ends(Pw, phi, tol=self.tol)
        #P = M*phi + S*(1-phi)
        slabs.extend(rho = [real(phi)], irho = [imag(phi)], w = Pw)
        #slabs.interface(self.interface.value)

    def start(self):
        return self.profile([0.], **self._fpars())[0]

    def end(self):
        return self.profile([self.thickness.value], **self._fpars())[0]

    def _fpars(self):
        kw = dict((k,getattr(self,k).value) for k in self._parameters)
        return  kw


class FunctionalMagnetism(BaseMagnetism):
    """
    Functional magnetism profile.

    Parameters:

        *profile* the profile function, suitably parameterized
        *tol* is the tolerance for considering values equal
        :class:`refl1d.magnetism.BaseMagnetism` parameters

    The profile function takes a depth vector *z* and returns a magnetism
    vector *rhoM*. For magnetic twist, return a pair of vectors (rhoM, thetaM).
    Constants can be returned using numpy.ones_like(z)*value.

    See :class:`FunctionalProfile` for a description of the the profile
    function.
    """
    RESERVED = ('profile','tol','name','extent', 'dead_below', 'dead_above',
                'interface_below', 'interface_above')
    magnetic = True
    def __init__(self, profile=None, tol=1e-3, name=None, **kw):
        if not name: name = profile.__name__
        if profile is None: raise TypeError("Need profile")
        # strip magnetism keywords from list of keywords
        magkw = dict((a,kw.pop(a)) for a in set(self.RESERVED)&set(kw.keys()))
        BaseMagnetism.__init__(self, name=name, **magkw)
        self.profile = profile
        self.tol = tol

        self._parameters = _set_vars(self, name, profile, kw, self.RESERVED)

    def parameters(self):
        parameters = BaseMagnetism.parameters(self)
        parameters.update( (k,getattr(self,k)) for k in self._parameters )
        return parameters

    def render(self, probe, slabs, thickness, anchor, sigma):
        Pw,Pz = slabs.microslabs(thickness)
        if len(Pw)== 0: return
        kw = dict((k,getattr(self,k).value) for k in self._parameters)
        #print kw
        P = self.profile(Pz,**kw)
        try: rhoM, thetaM = P
        except: rhoM, thetaM = P, Pz*0
        rhoM, thetaM = [asarray(v) for v in (rhoM,thetaM)]
        if rhoM.shape != Pz.shape or thetaM.shape != Pz.shape:
            raise TypeError("profile function '%s' did not return array rhoM(z)"
                            %self.profile.__name__)
        P = rhoM + thetaM*0.001j  # combine rhoM/thetaM so they can be merged
        Pw,P = util.merge_ends(Pw, P, tol=self.tol)
        rhoM,thetaM = P.real,P.imag*1000  # split out rhoM,thetaM again
        slabs.add_magnetism(anchor=anchor,
                            w=Pw,rhoM=rhoM,thetaM=thetaM,
                            sigma=sigma)

    def start(self):
        return self.profile([0.], **self._fpars())[0]

    def end(self):
        return self.profile([self.thickness.value], **self._fpars())[0]

    def _fpars(self):
        kw = dict((k,getattr(self,k).value) for k in self._parameters)
        return  kw

    def __repr__(self):
        return "FunctionalMagnetism(%s)"%self.name


def _set_vars(self, name, profile, kw, reserved):
    # Query profile function for the list of arguments
    vars = inspect.getargspec(profile)[0]
    #print "vars",vars
    if inspect.ismethod(profile): vars = vars[1:]  # Chop self
    vars = vars[1:]  # Chop z
    #print vars
    unused = [k for k in kw.keys() if k not in vars]
    if len(unused) > 0:
        raise TypeError("Profile got unexpected keyword argument '%s'"%unused[0])
    dups = [k for k in vars if k in reserved]
    if len(dups) > 0:
        raise TypeError("Profile has conflicting argument %r"%dups[0])
    for k in vars: kw.setdefault(k,0)
    for k,v in kw.items():
        setattr(self,k,Parameter.default(v,name=name+" "+k))

    return vars
