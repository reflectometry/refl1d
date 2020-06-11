"""
Load garefl models into refl1d.

The models themselves don't need to be modified.  See the garefl documentation
for setting up the model.

One extension provided to refl1d that is not available in garefl is the use
of penalty values in the constraints.  The model constraints is able to set::

    fit[0].penalty = FIT_REJECT_PENALTY + distance

Here, *distance* is the distance to the valid region of the search space so
that any fitter that gets lost in a penalty region can more quickly return
to the valid region.  Any penalty value above *FIT_REJECT_PENALTY* will
suppress the evaluation of the model at that point during the fit.

Consider a model with layers (Si | Au | FeNi | air) and the constraint that
d_Au + d_FeNi < 200 A.  The constraints function would be written something
like::

    double excess = fit[0].m.d[1] + fit[0].m.d[2] - 200;
    fit[0].penalty = excess > 0 ? excess*excess+FIT_REJECT_PENALTY : 0.;

Then, if the fit algorithm proposes a value such as Au=125, FeNi=90, the
excess will be 15, and the penalty will be FIT_REJECT_PENALTY+225.

You can use penalties less than *FIT_REJECT_PENALTY*, but these should
correspond to the negative log likelihood of seeing that constraint value
within the model in order for the MCMC uncertainty analysis to work correctly.
*FIT_REJECT_PENALTY* is set to 1e6, which should be high enough that it
doesn't perturb the fit.
"""
__all__ = ["load"]

import sys
import os
from os import getpid
from ctypes import CDLL, c_int, c_double, c_void_p, c_char_p, byref
from threading import current_thread

import numpy as np
from numpy import empty, zeros, array

from bumps.parameter import Parameter, to_dict
from bumps.fitproblem import FitProblem

from .probe import QProbe, PolarizedNeutronQProbe
from .experiment import Experiment
from .model import Stack
from .profile import Microslabs
from .material import SLD, Vacuum

if sys.version_info[0] >= 3:
    def tostr(s):
        return s.decode('ascii')
else:
    def tostr(s):
        return s

def trace(fn):
    """simple function trace function"""
    return fn  # Comment this to turn on tracing
    def wrapper(*args, **kw):
        print("entering %s for thread %s:%s"
              %(fn.func_name, getpid(), current_thread()))
        ret = fn(*args, **kw)
        print("leaving %s for thread %s:%s"
              %(fn.func_name, getpid(), current_thread()))
        return ret
    return wrapper

def load(modelfile, probes=None):
    """
    Load a garefl model file as an experiment.

    *modelfile* is a model.so file created from setup.c.

    *probes* is a list of datasets to fit to the models in the model file, or
    None if the model file provides its own data.
    """
    M = experiment(modelfile, probes)
    constraints = M[0]._get_penalty
    if len(M) > 1:
        return FitProblem(M, constraints=constraints)
    else:
        return FitProblem(M[0], constraints=constraints)

def experiment(modelfile, probes=None):
    setup = GareflModel(modelfile)
    if probes:
        if len(probes) != setup.num_models:
            raise ValueError("Number of datasets must match number of models")
        M = [GareflExperiment(setup, k, probe=probes[k])
             for k in range(setup.num_models)]
    else:
        M = [GareflExperiment(setup, k) for k in range(setup.num_models)]
    names = setup.par_names()
    low, high = setup.par_bounds()
    value = setup.par_values()
    pars = [Parameter(v, name=s, bounds=(L, H))
            for v, s, L, H in zip(value, names, low, high)]
    M[0]._pars = pars
    return M

NOTHING=Vacuum()
NOTHING.name = ''

class GareflExperiment(Experiment):
    def __init__(self, model, index, dz=1, step_interfaces=None, probe=None):
        self.model = model
        self.index = index
        if probe is None:
            probe = model.get_probe(index)
        else:
            model.set_probe(probe)
        self.probe = probe
        self.sample = Stack([NOTHING, NOTHING])
        self.sample[0].interface.fittable = False
        self.step_interfaces = True
        self._slabs = Microslabs(1, dz=dz)
        self._cache = {}  # Cache calculated profiles/reflectivities
        self._pars = None
        self.roughness_limit = 2.35
        self._substrate = SLD(name='substrate', rho=0)
        self._surface = SLD(name='surface', rho=0)
        self._name = None
        self.interpolation = 0

    def parameters(self):
        return self._pars

    def to_dict(self):
        return to_dict({
            'type': type(self).__name__,
            'dll_path': self.model._dll_path,
            'index': self.index,
            'parameters': self.parameters(),
        })

    def _render_slabs(self):
        """
        Build a slab description of the model from the individual layers.
        """
        key = 'rendered'
        if key not in self._cache:

            if self._pars is not None:
                pvec = array([p.value for p in self._pars], 'd')
                self._chisq = self.model.update_model(pvec, forced=True)

            self._slabs.clear()
            w, rho, irho, rhoM, thetaM = self.model.get_profile(self.index)
            rho, irho, rhoM = 1e6*rho, 1e6*irho, 1e6*rhoM # remove zeros
            self._slabs.extend(w=w, rho=rho[None, :], irho=irho[None, :])
            # TODO: What about rhoM, thetaM

            # Set values for the Fresnel-normalized reflectivity plot
            self._substrate.rho.value = rho[0]
            self._substrate.irho.value = irho[0]
            self._surface.rho.value = rho[-1]
            self._surface.irho.value = irho[-1]
            self._cache[key] = True
        return self._slabs

    def _get_penalty(self):
        """
        Update the model if necessary and return the penalty value for the point.
        """
        self._render_slabs()
        return self.model.get_penalty()

    def amplitude(self, resolution=True):
        """
        Calculate reflectivity amplitude at the probe points.
        """
        raise NotImplementedError("amplitude not yet available from garefl")

    def reflectivity(self, resolution=True, interpolation=0):
        """
        Calculate predicted reflectivity.
        """
        key = 'reflectivity'
        if key not in self._cache:
            self._render_slabs()  # Force recacluation
            if self.probe.polarized:
                Q, Rmm = self.model.get_reflectivity(self.index, 0)
                Q, Rmp = self.model.get_reflectivity(self.index, 1)
                Q, Rpm = self.model.get_reflectivity(self.index, 2)
                Q, Rpp = self.model.get_reflectivity(self.index, 3)
                self._cache[key] = Q, (Rmm, Rmp, Rpm, Rpp)
            else:
                Q, R = self.model.get_reflectivity(self.index, 0)
                self._cache[key] = Q, R
        return self._cache[key]

    def output_model(self):
        self.model.output_model()

class GareflModel(object):
    def __init__(self, path):
        self._dll_path = os.path.abspath(path)
        self._load_dll()
        self._setup_model()

    @trace
    def _load_dll(self):
        dll = CDLL(self._dll_path)
        dll.ex_get_data.restype = c_char_p
        dll.ex_set_data.restype = c_int
        dll.setup_models.restype = c_void_p
        dll.ex_par_name.restype = c_char_p
        dll.ex_get_penalty.restype = c_double
        self.dll = dll
        self.num_models = 0

    @trace
    def _setup_model(self):
        if self.num_models:
            raise RuntimeError("Model already loaded")
        MODELS = c_int()
        self.models = c_void_p(self.dll.setup_models(byref(MODELS)))
        self.num_models = MODELS.value
        self.num_pars = self.dll.ex_npars(self.models)
        lo, hi = self._par_bounds()
        small = np.max(np.vstack((abs(lo), abs(hi))), axis=0) < 1e-3
        self.scale = np.where(small, 1e6, 1)

        # TODO: better way to force recalc on load
        self.update_model(self.par_values())

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
    def update_model(self, p, weighted=1, approximate_roughness=0, forced=False):
        p = p/self.scale
        #assert p.flags.aligned and (p.flags.c_contiguous or p.flags.f_contiguous)
        #assert p.size == self.num_pars
        self.dll.ex_set_pars(self.models, p.ctypes)
        chisq = self.dll.ex_update_models(self.models, self.num_models,
                                          weighted, approximate_roughness,
                                          int(forced))
        return chisq

    @trace
    def get_probe(self, k):
        """
        Return a probe for an individual garefl model.
        """
        data = [self._getdata(k, xs) for xs in range(4)]
        if any(d is not None for d in data[1:]):
            probe = PolarizedNeutronQProbe(xs=data)
        else:
            probe = data[0]
        return probe

    def _getdata(self, k, xs):
        """
        Convert a single model into a probe
        """
        n = self.dll.ex_ndata(self.models, k, xs)
        if n == 0:
            return None
        data = empty((n, 4), 'd')
        filename = tostr(self.dll.ex_get_data(self.models, k, xs, data.ctypes))
        Q, dQ, R, dR = data.T
        probe = QProbe(Q, dQ, data=(R, dR), name=filename)
        return probe

    @trace
    def set_probe(self, k, probe):
        """
        Return a probe for an individual garefl model.
        """
        if probe.polarized:
            for xs, probe_xs in enumerate(probe.xs):
                self._setdata(k, xs, probe_xs)
        else:
            self._setdata(k, 0, probe)

    def _setdata(self, k, xs, probe):
        if probe is not None:
            n = probe.Q
            data = empty((n, 4), dtype='d', order='F')
            data[:, 0] = probe.Q
            data[:, 1] = probe.dQ
            data[:, 2] = probe.R
            data[:, 3] = probe.dR
        else:
            n = 0
            data = empty((n, 4), 'd')
        result = self.dll.ex_set_data(self.models, k, xs, n, data.ctypes)
        if result < 0:
            raise RuntimeError("unable to create data in garefl")

    @trace
    def get_profile(self, k):
        n = self.dll.ex_nprofile(self.models, k)
        w, rho, irho, rhoM, thetaM = [zeros(n, 'd') for _ in range(5)]
        self.dll.ex_get_profile(self.models, k, w.ctypes,
                                rho.ctypes, irho.ctypes,
                                rhoM.ctypes, thetaM.ctypes)
        return w[::-1], rho[::-1], irho[::-1], rhoM[::-1], thetaM[::-1]

    @trace
    def get_reflectivity(self, k, xs):
        n = self.dll.ex_ncalc(self.models, k)
        Q, R = empty(n, 'd'), empty(n, 'd')
        self.dll.ex_get_reflectivity(self.models, k, xs,
                                     Q.ctypes, R.ctypes)
        return Q, R

    @trace
    def get_penalty(self):
        #print "penalty", self.dll.ex_get_penalty(self.models), self.par_values()
        return self.dll.ex_get_penalty(self.models)

    @trace
    def output_model(self):
        """
        Run the output_model function
        """
        return self.dll.ex_output_model(self.models)

    @trace
    def par_names(self):
        return [tostr(self.dll.ex_par_name(self.models, i))
                for i in range(self.num_pars)]

    @trace
    def par_bounds(self):
        return self._par_bounds()*self.scale

    @trace
    def _par_bounds(self):
        lo, hi = empty(self.num_pars, 'd'), empty(self.num_pars, 'd')
        self.dll.ex_par_bounds(self.models, lo.ctypes, hi.ctypes)
        return lo, hi

    @trace
    def par_values(self):
        p = empty(self.num_pars, 'd')
        self.dll.ex_par_values(self.models, p.ctypes)
        return p*self.scale
