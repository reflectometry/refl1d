"""
Load garefl models into refl1d
"""
__all__ = ["load"]

import os
from ctypes import CDLL, c_int, c_void_p, c_char_p, byref
from threading import current_thread
from os import getpid

import numpy
from numpy import empty, zeros, array

from bumps.parameter import Parameter
from bumps.fitproblem import FitProblem, MultiFitProblem

from .probe import QProbe, PolarizedNeutronQProbe
from .experiment import Experiment
from .model import Stack
from .profile import Microslabs
from .material import SLD, Vacuum

def trace(fn):
    """simple function trace function"""
    return fn  # Comment this to turn on tracing
    def wrapper(*args, **kw):
        print "entering",fn.func_name,"from",current_thread(),getpid()
        ret = fn(*args, **kw)
        print "leaving",fn.func_name,"from",current_thread(),getpid()
        return ret
    return wrapper

def load(modelfile):
    M = experiment(modelfile)
    if len(M) > 1:
        return MultiFitProblem(M)
    else:
        return FitProblem(M[0])

def experiment(modelfile):
    setup = GareflModel(modelfile)
    M = [GareflExperiment(setup, k)
         for k in range(setup.num_models)]
    names = setup.par_names()
    low,high = setup.par_bounds()
    value = setup.par_values()
    pars = [Parameter(v, name=s, bounds=(L,H))
            for v,s,L,H in zip(value,names,low,high)]
    M[0]._pars = pars
    return M

NOTHING=Vacuum()
NOTHING.name = ''

class GareflExperiment(Experiment):
    def __init__(self, model, index, dz=1, step_interfaces=None):
        self.model = model
        self.index = index
        self.probe = model.get_probe(index)
        self.sample = Stack([NOTHING,NOTHING])
        self.sample[0].interface.fittable = False
        self.step_interfaces = True
        self._slabs = Microslabs(1, dz=dz)
        self._cache = {}  # Cache calculated profiles/reflectivities
        self._pars = None
        self.roughness_limit = 2.35
        self._substrate = SLD(name='substrate',rho=0)
        self._surface = SLD(name='surface',rho=0)
        self._name = None

    def parameters(self):
        return self._pars

    def _render_slabs(self):
        """
        Build a slab description of the model from the individual layers.
        """
        key = 'rendered'
        if key not in self._cache:

            if self._pars is not None:
                pvec = array([p.value for p in self._pars], 'd')
                self._chisq = self.model.update_model(pvec)

            self._slabs.clear()
            w,rho,irho,rhoM,thetaM = self.model.get_profile(self.index)
            rho,irho,rhoM = 1e6*rho,1e6*irho,1e6*rhoM # remove zeros
            self._slabs.extend(w=w,rho=rho[None,:],irho=irho[None,:])
            # TODO: What about rhoM, thetaM

            # Set values for the Fresnel reflectivity plot
            self._substrate.rho.value = rho[0]
            self._substrate.irho.value = irho[0]
            self._surface.rho.value = rho[-1]
            self._surface.irho.value = irho[-1]
            self._cache[key] = True
        return self._slabs

    def amplitude(self, resolution=True):
        """
        Calculate reflectivity amplitude at the probe points.
        """
        raise NotImplementedError("amplitude not available from garefl")

    def reflectivity(self, resolution=True):
        """
        Calculate predicted reflectivity.
        """
        key = 'reflectivity'
        if key not in self._cache:
            self._render_slabs()  # Force recacluation
            if self.probe.polarized:
                Q,Rmm = self.model.get_reflectivity(self.index, 0)
                Q,Rmp = self.model.get_reflectivity(self.index, 1)
                Q,Rpm = self.model.get_reflectivity(self.index, 2)
                Q,Rpp = self.model.get_reflectivity(self.index, 3)
                self._cache[key] = Q,(Rmm,Rmp,Rpm,Rpp)
            else:
                Q,R = self.model.get_reflectivity(self.index, 0)
                self._cache[key] = Q,R
        return self._cache[key]


class GareflModel(object):
    def __init__(self, path):
        self._dll_path = os.path.abspath(path)
        self._load_dll()
        self._setup_model()

    @trace
    def _load_dll(self):
        dll = CDLL(self._dll_path)
        dll.ex_get_data.restype = c_char_p
        dll.setup_models.restype = c_void_p
        dll.ex_par_name.restype = c_char_p
        self.dll = dll
        self.num_models = 0

    @trace
    def _setup_model(self):
        if self.num_models:
            raise RuntimeError("Model already loaded")
        MODELS = c_int()
        self.models = c_void_p(self.dll.setup_models(byref(MODELS)))
        self.num_models = MODELS.value
        lo, hi = self._par_bounds()
        small = numpy.max(numpy.vstack((abs(lo),abs(hi))),axis=0)<1e-4
        self.scale = numpy.where(small, 1e6, 1)

    # Pickle protocol doesn't support ctypes linkage; reload the
    # module on the other side.
    def __getstate__(self):
        return self._dll_path
    def __setstate__(self, state):
        self._dll_path = state
        self._load_dll()
        self._setup_model()


    def clear_model(self):
        if self.num_models:
            self.dll.ex_fit_destroy(self.models)
            self.num_models = 0

    @trace
    def update_model(self, p, weighted=1, approximate_roughness=0):
        p = p/self.scale
        self.dll.ex_set_pars(self.models, p.ctypes.data)
        chisq = self.dll.ex_update_models(self.models, self.num_models,
                                       weighted, approximate_roughness)
        return chisq

    @trace
    def get_probe(self, k):
        """
        Return a probe for an individual garefl model.
        """
        data = [self._getdata(k, xs) for xs in range(4)]
        if any(d!=None for d in data[1:]):
            probe = PolarizedNeutronQProbe(xs=data)
        else:
            probe = data[0]
        return probe

    def _getdata(self, k, xs):
        """
        Convert a single model into a probe
        """
        n = self.dll.ex_ndata(self.models, k, xs);
        if n == 0: return None
        data = empty((n,4),'d')
        filename = self.dll.ex_get_data(self.models, k, xs, data.ctypes.data)
        Q,dQ,R,dR = data.T
        probe = QProbe(Q,dQ,data=(R,dR))
        probe.filename = filename
        return probe

    @trace
    def get_profile(self, k):
        n = self.dll.ex_nprofile(self.models, k)
        w, rho,irho,rhoM,thetaM = [zeros(n,'d') for _ in range(5)]
        self.dll.ex_get_profile(self.models, k, w.ctypes.data,
                             rho.ctypes.data, irho.ctypes.data,
                             rhoM.ctypes.data, thetaM.ctypes.data)
        return w[::-1], rho[::-1], irho[::-1], rhoM[::-1], thetaM[::-1]

    @trace
    def get_reflectivity(self, k, xs):
        n = self.dll.ex_ncalc(self.models, k)
        Q, R = empty(n,'d'), empty(n,'d')
        self.dll.ex_get_reflectivity(self.models, k, xs,
                                     Q.ctypes.data, R.ctypes.data)
        return Q, R

    @trace
    def par_names(self):
        n = self.dll.ex_npars(self.models)
        return [self.dll.ex_par_name(self.models, i) for i in range(n)]

    @trace
    def par_bounds(self):
        return self._par_bounds()*self.scale

    @trace
    def _par_bounds(self):
        n = self.dll.ex_npars(self.models)
        lo,hi = empty(n,'d'), empty(n,'d')
        self.dll.ex_par_bounds(self.models, lo.ctypes.data, hi.ctypes.data)
        return lo, hi

    @trace
    def par_values(self):
        n = self.dll.ex_npars(self.models)
        p = empty(n,'d')
        self.dll.ex_par_values(self.models, p.ctypes.data)
        return p*self.scale
