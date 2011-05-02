# This program is in the public domain
# Author: Paul Kienzle

"""
Experiment definition

An experiment combines the sample definition with a measurement probe
to create a fittable reflectometry model.
"""

from math import log, pi, log10, ceil, floor
import shutil
import os
import traceback

import numpy
from .reflectivity import reflectivity_amplitude as reflamp
#print "Using pure python reflectivity calculator"
#from .abeles import refl as reflamp
from . import material, profile
from .fresnel import Fresnel
from .mystic.parameter import Parameter


def plot_sample(sample, instrument=None, roughness_limit=0):
    """
    Quick plot of a reflectivity sample and the corresponding reflectivity.
    """
    if instrument == None:
        from .probe import NeutronProbe
        probe = NeutronProbe(T=numpy.arange(0,5,0.05), L=5)
    else:
        probe = instrument.simulate()
    experiment = Experiment(sample=sample, probe=probe,
                            roughness_limit=roughness_limit)
    experiment.plot()

class ExperimentBase(object):
    def format_parameters(self):
        import mystic.parameter
        p = self.parameters()
        print mystic.parameter.format(p)

    def update_composition(self):
        """
        When the model composition has changed, we need to lookup the
        scattering factors for the new model.  This is only needed
        when an existing chemical formula is modified; new and
        deleted formulas will be handled automatically.
        """
        self._probe_cache.reset()
        self.update()

    def update(self):
        """
        Called when any parameter in the model is changed.

        This signals that the entire model needs to be recalculated.
        """
        # if we wanted to be particularly clever we could predefine
        # the optical matrices and only adjust those that have changed
        # as the result of a parameter changing.   More trouble than it
        # is worth, methinks.
        self._cache = {}

    def residuals(self):
        if self.probe.R is None:
            raise ValueError("No data from which to calculate residuals")
        if 'residuals' not in self._cache:
            _,R = self.reflectivity()
            resid = (self.probe.R - R)/self.probe.dR
            self._cache['residuals'] = resid

        return self._cache['residuals']

    def numpoints(self):
        return len(self.probe.Q)

    def nllf(self):
        """
        Return the -log(P(data|model)).

        Using the assumption that data uncertainty is uncorrelated, with
        measurements normally distributed with mean R and variance dR**2,
        this is just sum( resid**2/2 + log(2*pi*dR**2)/2 ).

        The current version drops the constant term, sum(log(2*pi*dR**2)/2).
        """
        #if 'nllf_scale' not in self._cache:
        #    if self.probe.dR is None:
        #        raise ValueError("No data from which to calculate nllf")
        #    self._cache['nllf_scale'] = numpy.sum(numpy.log(2*pi*self.probe.dR**2))
        # TODO: add sigma^2 effects back into nllf; only needs to be calculated
        # when dR changes, so maybe it belongs in probe.
        return 0.5*numpy.sum(self.residuals()**2) # + self._cache['nllf_scale']

    def plot_reflectivity(self, show_resolution=False, view=None):

        Q,R = self.reflectivity()
        self.probe.plot(theory=(Q,R),
                        substrate=self._substrate, surface=self._surface,
                        view=view)

        if show_resolution:
            import pylab
            Q,R = self.reflectivity(resolution=False)
            pylab.plot(Q,R,':g',hold=True)

    def plot(self):
        import pylab
        pylab.subplot(211)
        self.plot_profile()
        pylab.subplot(212)
        self.plot_reflectivity()


    def resynth_data(self):
        """Resynthesize data with noise from the uncertainty estimates."""
        self.probe.resynth_data()
    def restore_data(self):
        """Restore original data after resynthesis."""
        self.probe.restore_data()
    def write_data(self, filename, **kw):
        """Save simulated data to a file"""
        self.probe.write_data(filename, **kw)
    def simulate_data(self, noise=2):
        """
        Simulate a random data set for the model

        **Parameters:**

        *noise* = 2 : float | %
            Percentage noise to add to the data.
        """
        _,R = self.reflectivity(resolution=True)
        dR = 0.01*noise*R
        self.probe.simulate_data(R,dR)

class Experiment(ExperimentBase):
    """
    Theory calculator.  Associates sample with data, Sample plus data.
    Associate sample with measurement.

    The model calculator is specific to the particular measurement technique
    that was applied to the model.

    Measurement properties:

        *probe* is the measuring probe

    Sample properties:

        *sample* is the model sample
        *roughness_limit* limits the roughness based on layer thickness
        *dz* minimum step size for computed profile steps in Angstroms.
        *dA* discretization condition for computed profiles.

    The *roughness_limit* value should be reasonably large (e.g., 2.5 or above)
    to make sure that the Nevot-Croce reflectivity calculation matches the
    calculation of the displayed profile.  Use a value of 0 if you want no
    limits on the roughness,  but be aware that the displayed profile may
    not reflect the actual scattering densities in the material.

    The *dz* step size sets the size of the slabs for non-uniform profiles.
    Using the relation d = 2 pi / Q_max,  we use a default step size of d/20
    rounded to two digits, with 5 |Ang| as the maximum default.  For
    simultaneous fitting you may want to set *dz* explicitly using to
    round(pi/Q_max/10,1) so that all models use the same step size.

    The *dA* condition measures the uncertainty in scattering materials
    allowed when combining the steps of a non-uniform profile into slabs.
    Specifically, the area of the box containing the minimum and the
    maximum of the non-uniform profile within the slab will be smaller
    than *dA*.  A *dA* of 10 gives coarse slabs.  If *dA* is not provided
    then each profile step forms its own slab.
    """
    def __init__(self, sample=None, probe=None,
                 roughness_limit=2.5, dz=None, dA=None):
        self.sample = sample
        self._substrate=self.sample[0].material
        self._surface=self.sample[-1].material
        self.probe = probe
        self.roughness_limit = roughness_limit
        if dz is None:
            dz = nice((2*pi/probe.Q.max())/20)
            if dz > 5: dz = 5
        self.dA = dA
        self._slabs = profile.Microslabs(len(probe), dz=dz)
        self._probe_cache = material.ProbeCache(probe)
        self._cache = {}  # Cache calculated profiles/reflectivities

    def parameters(self):
        return dict(sample=self.sample.parameters(),
                    probe=self.probe.parameters())

    def _render_slabs(self):
        """
        Build a slab description of the model from the individual layers.
        """
        key = 'rendered'
        if key not in self._cache:
            self._slabs.clear()
            self.sample.render(self._probe_cache, self._slabs)
            if self.dA is not None:
                self._slabs.contract_profile(self.dA)
                self._slabs.smooth_interfaces(self.dA)
            self._cache[key] = True
        return self._slabs

    def _reflamp(self):
        #calc_q = self.probe.calc_Q
        #return calc_q,calc_q
        key = 'calc_r'
        if key not in self._cache:
            slabs = self._render_slabs()
            w = slabs.w
            rho,irho = slabs.rho, slabs.irho
            #sigma = slabs.limited_sigma(limit=self.roughness_limit)
            sigma = slabs.sigma
            calc_q = self.probe.calc_Q
            calc_r = reflamp(-calc_q/2, depth=w, rho=rho, irho=irho,
                             sigma=sigma)
            if numpy.isnan(calc_r).any():
                print "w",w
                print "rho",rho
                print "irho",irho
                print "sigma",sigma
                print "kz",self.probe.calc_Q/2
                print "R",abs(calc_r**2)
                from .mystic import parameter
                pars = parameter.unique(self.parameters())
                fitted = parameter.varying(pars)
                print parameter.summarize(fitted)
                print "==="
            self._cache[key] = calc_q,calc_r
            if numpy.isnan(calc_q).any(): print "calc_Q contains NaN"
            if numpy.isnan(calc_r).any(): print "calc_r contains NaN"
        return self._cache[key]

    def amplitude(self, resolution=True):
        """
        Calculate reflectivity amplitude at the probe points.
        """
        key = ('amplitude',resolution)
        if key not in self._cache:
            calc_q,calc_r = self._reflamp()
            r_real = self.probe.apply_beam(calc_q, calc_r.real, resolution=resolution)
            r_imag = self.probe.apply_beam(calc_q, calc_r.imag, resolution=resolution)
            r = r_real + 1j*r_imag
            self._cache[key] = self.probe.Q, r
        return self._cache[key]


    def reflectivity(self, resolution=True):
        """
        Calculate predicted reflectivity.

        If *resolution* is true include resolution effects.

        If *beam* is true, include absorption and intensity effects.
        """
        key = ('reflectivity',resolution)
        if key not in self._cache:
            calc_q, calc_r = self._reflamp()
            calc_R = abs(calc_r)**2
            if numpy.isnan(calc_R).any():
                print "calc_r contains NaN"
                slabs = self._slabs
                #print "w",slabs.w
                #print "rho",slabs.rho
                #print "irho",slabs.irho
                #print "sigma",slabs.sigma
            Q,R = self.probe.apply_beam(calc_q, calc_R, resolution=resolution)
            #Q,R = self.probe.Qo,self.probe.R
            self._cache[key] = Q,R
            if numpy.isnan(R).any(): print "apply_beam causes NaN"
        return self._cache[key]

    def fresnel(self):
        """
        Calculate the fresnel reflectivity for the model.
        """
        slabs = self._render_slabs()
        rho,irho = slabs.rho, slabs.irho
        #sigma = slabs.limited_sigma(limit=self.roughness_limit)
        #sigma = slabs.sigma
        sigma = [0] # Don't do roughness
        f = Fresnel(rho=rho[0,0], irho=irho[0,0], sigma=sigma[0],
                    Vrho=rho[0,-1], Virho=irho[0,-1])
        return f(self.probe.Q)

    def smooth_profile(self,dz=0.1):
        """
        Compute a density profile for the material.

        If *dz* is not given, use *dz* = 1 A.
        """
        key = 'smooth_profile', dz
        if key not in self._cache:
            slabs = self._render_slabs()
            prof = slabs.smooth_profile(dz=dz,
                                        roughness_limit=self.roughness_limit)
            self._cache[key] = prof
        return self._cache[key]

    def step_profile(self):
        """
        Compute a scattering length density profile
        """
        key = 'step_profile'
        if key not in self._cache:
            slabs = self._render_slabs()
            prof = slabs.step_profile()
            self._cache[key] = prof
        return self._cache[key]

    def slabs(self):
        """
        Return the slab thickness, roughness, rho, irho for the
        rendered model.

        .. Note::
             Roughness is for the top of the layer.
        """
        slabs = self._render_slabs()
        return (slabs.w, numpy.hstack((0,slabs.sigma)),
                slabs.rho[0], slabs.irho[0])

    def save(self, basename):
        self.save_profile(basename)
        self.save_staj(basename)
        self.save_refl(basename)

    def save_profile(self, basename):
        # Slabs
        A = numpy.array(self.slabs())
        fid = open(basename+"-slabs.dat","w")
        fid.write("# %17s %20s %20s %20s\n"%("thickness","roughness",
                                              "rho (1e-6/A2)","irho (1e-6/A2)"))
        numpy.savetxt(fid, A.T, fmt="%20.15g")
        fid.close()

        # Step profile
        A = numpy.array(self.step_profile())
        fid = open(basename+"-steps.dat","w")
        fid.write("# %10s %12s %12s\n"%("z","rho (1e-6/A2)","irho (1e-6/A2)"))
        numpy.savetxt(fid, A.T, fmt="%12.8f")
        fid.close()

        # Smooth profile
        A = numpy.array(self.smooth_profile())
        fid = open(basename+"-profile.dat","w")
        fid.write("# %10s %12s %12s\n"%("z","rho (1e-6/A2)","irho (1e-6/A2)"))
        numpy.savetxt(fid, A.T, fmt="%12.8f")
        fid.close()

    def save_refl(self, basename):
        # Reflectivity
        theory = self.reflectivity()[1]
        A = numpy.array((self.probe.Q,self.probe.dQ,self.probe.R,self.probe.dR,
                         theory))
        fid = open(basename+"-refl.dat","w")
        fid.write("# %17s %20s %20s %20s %20s\n"%("Q (1/A)","dQ (1/A)",
                                                   "R", "dR", "theory"))
        numpy.savetxt(fid, A.T, fmt="%20.15g")
        fid.close()

    def save_staj(self, basename):
        from .stajconvert import save_mlayer
        try:
            save_mlayer(self, basename+".staj")
            probe = self.probe
            datafile = os.path.join(os.path.dirname(basename),probe.filename)
            fid = open(datafile,"w")
            fid.write("# Q R dR\n")
            numpy.savetxt(fid, numpy.vstack((probe.Q,probe.R,probe.dR)).T)
            fid.close()
        except:
            print "==== could not save staj file ===="
            traceback.print_exc()


    def plot_profile(self):
        import pylab
        z,rho,irho = self.step_profile()
        pylab.plot(z,rho,':g',z,irho,':b')
        z,rho,irho = self.smooth_profile()
        pylab.plot(z,rho,'-g',z,irho,'-b', hold=True)
        pylab.legend(['rho','irho'])
        pylab.xlabel('depth (A)')
        pylab.ylabel('SLD (10^6 inv A**2)')


class MixedExperiment(ExperimentBase):
    """
    Support composite sample reflectivity measurements.

    Sometimes the sample you are measuring is not uniform.
    For example, you may have one portion of you polymer
    brush sample where the brushes are close packed and able
    to stay upright, whereas a different section of the sample
    has the brushes lying flat.  Constructing two sample
    models, one with brushes upright and one with brushes
    flat, and adding the reflectivity incoherently, you can
    then fit the ratio of upright to flat.

    *samples* the layer stacks making up the models
    *ratio* a list of parameters, such as [3,1] for a 3:1 ratio
    *probe* the measurement to be fitted or simulated

    Statistics such as the cost functions for the individual
    profiles can be accessed from the underlying experiments
    using composite.parts[i] for the various samples.
    """
    def __init__(self, samples=None, ratio=None, probe=None, **kw):
        self.samples = samples
        self.probe = probe
        self.ratio = [Parameter.default(r, name="ratio %d"%i)
                      for i,r in enumerate(ratio)]
        self.parts = [Experiment(s,probe,**kw) for s in samples]
        self._substrate=self.samples[0][0].material
        self._surface=self.samples[0][-1].material
        self._cache = {}

    def update(self):
        self._cache = {}
        for p in self.parts: p.update()
        
    def parameters(self):
        return dict(samples = [s.parameters() for s in self.samples],
                    ratio = self.ratio,
                    probe = self.probe.parameters(),
                    )

    def reflectivity(self, resolution=True, beam=True):
        """
        Calculate predicted reflectivity.

        If *resolution* is true include resolution effects.

        If *beam* is true, include absorption and intensity effects.
        """
        f = numpy.array([r.value for r in self.ratio],'d')
        f /= numpy.sum(f)
        Qs,Rs = zip(*[p.reflectivity() for p in self.parts])
        Q = Qs[0]
        R = f*numpy.array(Rs).T
        return Q, numpy.sum(R,axis=1)

    def plot_profile(self):
        import pylab
        f = numpy.array([r.value for r in self.ratio],'d')
        f /= numpy.sum(f)
        held = pylab.hold()
        for p in self.parts:
            p.plot_profile()
            pylab.hold(True)
        pylab.hold(held)


def nice(v, digits = 2):
    """Fix v to a value with a given number of digits of precision"""
    if v == 0.: return v
    sign = v/abs(v)
    place = floor(log10(abs(v)))
    scale = 10**(place-(digits-1))
    return sign*floor(abs(v)/scale+0.5)*scale
