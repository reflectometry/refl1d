"""
Inhomogeneous samples

In the presence of samples with short range order on scale of the coherence
length of the probe in the plane, but long range disorder following some
distribution of parameter values, the reflectivity can be computed from
a weighted incoherent sum of the reflectivities for different values of
the parameter.

DistristributionExperiment allows the model to be computed for a single
varying parameter.  Multi-parameter dispersion models are not available.
"""

from bumps.parameter import Parameter
import numpy as np

from .experiment import ExperimentBase


class Weights(object):
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
    function signature f(x, a1, a2, ..., loc=0, scale=1).  The list *args*
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

    def __init__(self, edges=None, cdf=None, args=(), loc=None, scale=None, truncated=True):
        self.edges = np.asarray(edges)
        self.cdf = cdf
        self.truncated = truncated
        self.loc = Parameter.default(loc)
        self.scale = Parameter.default(scale)
        self.args = [Parameter.default(a) for a in args]

    def parameters(self):
        return {"args": self.args, "loc": self.loc, "scale": self.scale}

    def __iter__(self):
        # Find weights and normalize the sum to 1
        centers = (self.edges[:-1] + self.edges[1:]) / 2
        loc = self.loc.value
        scale = self.scale.value
        args = [p.value for p in self.args]
        cumulative_weights = self.cdf(self.edges, *args, loc=loc, scale=scale)
        if not self.truncated:
            cumulative_weights[0], cumulative_weights[-1] = 0, 1
        relative_weights = cumulative_weights[1:] - cumulative_weights[:-1]
        total_weight = np.sum(relative_weights)
        if total_weight == 0:
            return iter([])
        else:
            weights = relative_weights / total_weight
            idx = weights > 0
            return iter(zip(centers[idx], weights[idx]))


class DistributionExperiment(ExperimentBase):
    """
    Compute reflectivity from a non-uniform sample.

    *P* is the target parameter for the model, which takes on the values
    from *distribution* in the context of the *experiment*.  The result
    is the weighted sum of the theory curves after setting *P.value* to
    each distribution value. Clearly, *P* should not be a fitted parameter,
    but the remaining experiment parameters can be fitted, as can the
    parameters of the distribution.

    If *coherent* is true, then the reflectivity of the mixture is computed
    from the coherent sum rather than the incoherent sum.

    See :class:`Weights` for a description of how to set up the distribution.
    """

    def __init__(self, experiment=None, P=None, distribution=None, coherent=False):
        self.P = P
        self.distribution = distribution
        self.experiment = experiment
        self.probe = self.experiment.probe
        self.coherent = coherent
        self._substrate = self.experiment.sample[0].material
        self._surface = self.experiment.sample[-1].material
        self._cache = {}  # Cache calculated profiles/reflectivities
        self._name = None

    def parameters(self):
        return {
            "distribution": self.distribution.parameters(),
            "experiment": self.experiment.parameters(),
        }

    def reflectivity(self, resolution=True, interpolation=0):
        key = ("reflectivity", resolution, interpolation)
        if key not in self._cache:
            calc_R = 0
            for x, w in self.distribution:
                if w > 0:
                    self.P.value = x
                    self.experiment.update()
                    Qx, Rx = self.experiment._reflamp()
                    if self.coherent:
                        calc_R += w * Rx
                    else:
                        calc_R += w * abs(Rx) ** 2
            if self.coherent:
                calc_R = abs(calc_R) ** 2
            Q, R = self.probe.apply_beam(Qx, calc_R, resolution=resolution, interpolation=interpolation)
            self._cache[key] = Q, R
        return self._cache[key]

    def _max_P(self):
        x, w = zip(*self.distribution)
        idx = np.argmax(w)
        return x[idx]

    def smooth_profile(self, dz=1):
        """
        Compute a density profile for the material
        """
        key = "smooth_profile", dz
        if key not in self._cache:
            P = self._max_P()
            if self.P.value != P:
                self.P.value = P
                self.experiment.update()
            self._cache[key] = self.experiment.smooth_profile(dz=dz)
        return self._cache[key]

    def step_profile(self):
        """
        Compute a scattering length density profile
        """
        key = "step_profile"
        if key not in self._cache:
            P = self._max_P()
            if self.P.value != P:
                self.P.value = P
                self.experiment.update()
            self._cache[key] = self.experiment.step_profile()
        return self._cache[key]

    def plot_profile(self, plot_shift=0.0):
        from bumps.plotutil import auto_shift
        import matplotlib.pyplot as plt

        trans = auto_shift(plot_shift)
        z, rho, irho = self.step_profile()
        plt.plot(z, rho, "-g", z, irho, "-b", transform=trans)
        z, rho, irho = self.smooth_profile()
        plt.plot(z, rho, ":g", z, irho, ":b", transform=trans)
        plt.legend(["rho", "irho"])

    def plot_weights(self):
        import matplotlib.pyplot as plt

        x, w = zip(*self.distribution)
        plt.stem(x, 100 * np.array(w))
        plt.title("Weight distribution")
        plt.xlabel(self.P.name)
        plt.ylabel("Percentage")
