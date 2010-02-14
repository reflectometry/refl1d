# This program is in the public domain
# Author: Paul Kienzle

from math import log, pi, log10, ceil, floor
import numpy
from .abeles import refl
from . import material, profile

class ExperimentBase:
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
        if 'residuals' not in self._cache:
            Q,R = self.reflectivity()
            if self.probe.R is not None:
                self._cache['residuals'] = (self.probe.R - R)/self.probe.dR
            else:
                self._cache['residuals'] = numpy.zeros_like(Q)
        return self._cache['residuals']

    def nllf(self):
        """
        Return the -log(P(data|model)), scaled to P = 1 if the data
        exactly matches the model.

        This is just sum( resid**2/2 + log(2*pi*dR**2)/2 ) with the
        constant log(2 pi dR^2)/2 removed.
        """
        return numpy.sum(self.residuals()**2)/2

    def plot_reflectivity(self, show_resolution=False):
        Q,R = self.reflectivity()
        self.probe.plot(theory=(Q,R),
                        substrate=self.sample[0].material,
                        surface=self.sample[-1].material)
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

    
class Experiment(ExperimentBase):
    """
    Theory calculator.  Associates sample with data, Sample plus data.
    Associate sample with measurement.

    The model calculator is specific to the particular measurement technique
    that was applied to the model.

    Measurement properties::

        *probe* is the measuring probe

    Sample properties::

        *sample* is the model sample
        *roughness_limit* limits the roughness based on layer thickness
        *dz* step size for profile steps in Angstroms.

    The *roughness_limit* value should be reasonably large (e.g., 2.5 or above)
    to make sure that the Nevot-Croce reflectivity calculation matches the
    calculation of the displayed profile.  Use a value of 0 if you want no
    limits on the roughness,  but be aware that the displayed profile may
    not reflect the actual scattering densities in the material.
    
    The *dz* step size sets the size of the slabs for non-uniform profiles.
    Using the relation d = 2 pi / Q_max,  we use a default step size of d/20 
    rounded to two digits.  The maximum step size is 5 A.  For simultaneous 
    fitting you may want to set *dz* explicitly using 
    :function:`experiment.nice`  to nice(pi/Q_max/10) so that all models 
    use the same profile step size, but the same step size is not required.
    """
    def __init__(self, sample=None, probe=None,
                 roughness_limit=2.5, dz=None):
        self.sample = sample
        self.probe = probe
        self.roughness_limit = roughness_limit
        if dz is None: 
            dz = nice((2*pi/probe.Q.max())/20)
            if dz > 5: dz = 5
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
        if 'rendered' not in self._cache:
            self._slabs.clear()
            self.sample.render(self._probe_cache, self._slabs)
            self._cache['rendered'] = True

    def _reflamp(self):
        if 'calc_r' not in self._cache:
            self._render_slabs()
            w = self._slabs.w
            rho,irho = self._slabs.rho, self._slabs.irho
            sigma = self._slabs.limited_sigma(limit=self.roughness_limit)
            #sigma = self._slabs.sigma
            calc_r = refl(-self.probe.calc_Q/2,
                          depth=w, rho=rho, irho=irho, sigma=sigma)
            #print "w",w
            #print "rho",rho
            #print "irho",irho
            #print "sigma",sigma
            #print "kz",self.probe.calc_Q/2
            #print "R",abs(calc_r**2)
            self._cache['calc_r'] = calc_r
            if numpy.isnan(self.probe.calc_Q).any(): print "calc_Q contains NaN"
            if numpy.isnan(calc_r).any(): print "calc_r contains NaN"
        return self._cache['calc_r']

    def amplitude(self):
        """
        Calculate reflectivity amplitude at the probe points.
        """
        if 'amplitude' not in self._cache:
            calc_r = self._reflamp()
            r_real = self.probe.resolution(calc_r.real)
            r_imag = self.probe.resolution(calc_r.imag)
            r = r_real + 1j*r_imag
            self._cache['amplitude'] = self.probe.Q, r
        return self._cache['amplitude']


    def reflectivity(self, resolution=True, beam=True):
        """
        Calculate predicted reflectivity.

        If *resolution* is true include resolution effects.

        If *beam* is true, include absorption and intensity effects.
        """
        calc_r = self._reflamp()
        calc_R = abs(calc_r)**2
        if resolution:
            Q,R = self.probe.Q, self.probe.resolution(calc_R)
        else:
            Q,R = self.probe.calc_Q, calc_R
        if beam:
            R = self.probe.beam_parameters(R)
            if numpy.isnan(R).any(): print "beam contains NaN"
        return Q, R

    def smooth_profile(self,dz=1):
        """
        Compute a density profile for the material.
        
        If *dz* is not given, use *dz* = 1 A.
        """
        if ('smooth_profile',dz) not in self._cache:
            self._render_slabs()
            prof = self._slabs.smooth_profile(dz=dz,
                                              roughness_limit=self.roughness_limit)
            self._cache['smooth_profile',dz] = prof
        return self._cache['smooth_profile',dz]

    def step_profile(self):
        """
        Compute a scattering length density profile
        """
        if 'step_profile' not in self._cache:
            self._render_slabs()
            prof = self._slabs.step_profile()
            self._cache['step_profile'] = prof
        return self._cache['step_profile']

    def plot_profile(self):
        import pylab
        z,rho,irho = self.step_profile()
        pylab.plot(z,rho,'-g',z,irho,'-b')
        z,rho,irho = self.smooth_profile()
        pylab.plot(z,rho,':g',z,irho,':b', hold=True)
        pylab.legend(['rho','irho'])
        pylab.xlabel('depth (A)')
        pylab.ylabel('SLD (10^6 inv A**2)')

class CompositeExperiment(ExperimentBase):
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
    def __init__(self, samples=None, ratio=None, 
                 probe=None, roughness_limit=2.5):
        self.samples = samples
        self.ratio = [Parameter.default(r) for r in ratio]
        self.parts = [Experiment(s,probe) for s in samples]
        
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
        f = numpy.array(r.value for r in self.ratio)
        Qs,Rs = zip(*[p.reflectivity() for p in self.parts])
        Q = Qs[0]
        R = f/numpy.sum(f,axis=0)*numpy.array(Rs)        
        return Q, R


class Weights:
    """
    Parameterized distribution for use in DistributionExperiment.

    To support non-uniform experiments, we must bin the possible values
    for the parameter and compute the theory function for one parameter
    value per bin.  The weighted sum of the resulting theory functions
    is the value that we compare to the data.

    Performing this analysis requires a cumulative density function which
    can return the integrated value of the probability density from -inf
    to x.  The total density in each bin is then the difference between
    the cumulative densities at the edges.  If the distribution is wider
    than the range, then the tails need to be truncated and the bins
    reweighted to a total density of 1, or the tail density can be added
    to the first and last bins.  Weights of zero are not returned.  Note
    that if the tails are truncated, this may result in no weights being
    returned.

    The vector *edges* contains the bin edges for the distribution.  The
    function *cdf* returns the cumulative density function at the edges.
    The *cdf* function must implement the scipy.stats interface, with
    function signature f(x,a1,a2,...,loc=0,scale=1).  The list *args*
    defines the arguments a1, a2, etc.  The underlying parameters are
    available as args[i].  Similarly, *loc* and *scale* define the
    distribution center and width.  Use *truncated*=False if you want
    the distribution tails to be included in the weights.

    SciPy distribution D is used by specifying cdf=scipy.stats.D.cdf.
    Useful distributions include::

        norm      Gaussian distribution.
        halfnorm  Right half of a gaussian.
        triang    Triangle distribution from loc up to loc+args[0]*scale
                  and down to loc+scale.  Use loc=edges[0], scale=edges[-1]
                  and args=[0.5] to define a symmetric triangle in the range
                  of parameter P.
        uniform   Flat from loc to loc+scale. Use loc=edges[0], scale=edges[-1]
                  to define P as uniform over the range.
    """
    def __init__(self, edges=None, cdf=None,
                 args=None, loc=None, scale=None, truncated=True):
        self.edges = numpy.asarray(edges)
        self.cdf = cdf
        self.truncated = truncated
        self.loc = Parameter.default(loc)
        self.scale = Parameter.default(scale)
        self.args = [Parameter.default(a) for a in args]
    def parameters(self):
        return dict(args=self.args,loc=self.loc,scale=self.scale)
    def __iter__(self):
        # Find weights and normalize the sum to 1
        centers = (self.edges[:-1]+self.edges[1:])/2
        cumulative_weights = self.cdf(self.edges)
        if not self.truncated:
            self.edges[0],self.edges[-1] = 0,1
        relative_weights = cumulative_weights[1:] - cumulative_weights[:-1]
        total_weight = numpy.sum(relative_weights)
        if total_weight == 0:
            return iter([])
        else:
            weights = relative_weights / total_weight
            idx = weigths > 0
            return iter(zip(centers[idx], weights[idx]))

class DistributionExperiment(ExperimentBase):
    """
    Compute reflectivity from a non-uniform sample.

    The parameter *P* takes on the values from *distribution* in the
    context of *experiment*. Clearly, *P* should not be a fitted
    parameter, but the remaining experiment parameters can be fitted,
    as can the parameters of the distribution.

    See :class:`Weights` for a description of how to set up the distribution.
    """
    def __init__(self, experiment=None, P=None, distribution=None):
        self.P = P
        self.x = x
        self.distribution = distribution
        self.experiment = experiment
    def parameters(self):
        return dict(distribution=self.distribution.parameters(),
                    experiment=self.experiment.parameters())
    def reflectivity(self):
        R = 0
        for x,w in self.distribution:
            self.P.value = x
            self.experiment.update()
            Q,Rx = experiment.reflectivity()
            R += w*Rx
        if R == 0:
            Q = self.experiment.probe.Q
            return Q,numpy.zeros_like(Q)
        else:
            return Q,R

    def smooth_profile(self,P,dz=1):
        """
        Compute a density profile for the material
        """
        if self.P.value != P:
            self.P.value = P
            self.experiment.update()
        return self.experiment.smooth_profile(dz=dz)

    def step_profile(self, P):
        """
        Compute a scattering length density profile
        """
        if self.P.value != P:
            self.P.value = P
            self.experiment.update()
        return self.experiment.step_profile(dz=dz)

    def plot_profile(self, P):
        import pylab
        z,rho,irho = self.step_profile(P)
        pylab.plot(z,rho,'-g',z,irho,'-b')
        z,rho,irho = self.smooth_profile(P)
        pylab.plot(z,rho,':g',z,irho,':b')
        pylab.legend(['rho','irho'])

def nice(v, digits = 2):
    """Fix v to a value with a given number of digits of precision"""
    if v == 0.: return v
    sign = v/abs(v)
    place = floor(log10(abs(v)))
    scale = 10**(place-(digits-1))
    return sign*floor(abs(v)/scale+0.5)*scale
