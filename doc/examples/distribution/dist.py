
from refl1d.experiment import ExperimentBase

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
    distribution center and width.  Use *truncated=False* if you want
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
                 args=[], loc=None, scale=None, truncated=True):
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
        loc = self.loc.value
        scale = self.scale.value
        args = [p.value for p in self.args]
        cumulative_weights = self.cdf(self.edges, *args, loc=loc, scale=scale)
        if not self.truncated:
            cumulative_weights[0],cumulative_weights[-1] = 0,1
        relative_weights = cumulative_weights[1:] - cumulative_weights[:-1]
        total_weight = numpy.sum(relative_weights)
        if total_weight == 0:
            return iter([])
        else:
            weights = relative_weights / total_weight
            idx = weights > 0
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
        self.distribution = distribution
        self.experiment = experiment
        self.probe = self.experiment.probe
        self._substrate=self.experiment.sample[0].material
        self._surface=self.experiment.sample[-1].material
        self._cache = {}  # Cache calculated profiles/reflectivities
    def parameters(self):
        return dict(distribution=self.distribution.parameters(),
                    experiment=self.experiment.parameters())
    def reflectivity(self, **kw):
        Q = self.experiment.probe.Q
        R = 0*Q
        for x,w in self.distribution:
            if w>0:
                self.P.value = x
                self.experiment.update()
                Q,Rx = self.experiment.reflectivity(**kw)
                R += w*Rx
        return Q,R

    def _best_P(self):
        x,w = zip(*self.distribution)
        idx = numpy.argmax(w)
        return x[idx]

    def smooth_profile(self,dz=1):
        """
        Compute a density profile for the material
        """
        P = self._best_P()
        if self.P.value != P:
            self.P.value = P
            self.experiment.update()
        return self.experiment.smooth_profile(dz=dz)

    def step_profile(self):
        """
        Compute a scattering length density profile
        """
        P = self._best_P()
        if self.P.value != P:
            self.P.value = P
            self.experiment.update()
        return self.experiment.step_profile()

    def plot_profile(self):
        import pylab
        z,rho,irho = self.step_profile()
        pylab.plot(z,rho,'-g',z,irho,'-b')
        z,rho,irho = self.smooth_profile()
        pylab.plot(z,rho,':g',z,irho,':b')
        pylab.legend(['rho','irho'])

    def plot_weights(self):
        import pylab
        x,w = zip(*self.distribution)
        pylab.stem(x,100*numpy.array(w))
        pylab.title('Weight distribution')
        pylab.xlabel(self.P.name)
        pylab.ylabel('Percentage')
