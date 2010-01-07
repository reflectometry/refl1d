"""
Adaptors for fitting.

*WARNING* within models self.parameters() returns a tree of all possible
parameters associated with the model.  Within fit problems, self.parameters
is a list of fitted parameters only.
"""
from copy import deepcopy
from numpy import inf, nan, isnan
import numpy
from mystic import parameter, bounds as mbounds
import time

def preview(models=[], weights=None):
    """Preview the models in preparation for fitting"""
    if isinstance(models,(tuple,list)):
        problem = MultiFitProblem(models, weights=weights)
    else:
        problem = FitProblem(models)
    xo = [p.value for p in problem.parameters]
    result = Result(problem,xo)
    result.show()
    return problem

class FitBase:
    def __init__(self, models=[], weights=None, **kw):
        """Fit the models and show the results"""
        if isinstance(models,(tuple,list)):
            self.problem = MultiFitProblem(models, weights=weights)
        else:
            self.problem = FitProblem(models)
        self.time = time.clock()
        x = self.solve(**kw)
        self.problem.setp(x)
        result = Result(self.problem,x)
        print "time",time.clock()-self.time
        result.show()
    def solve(self, **kw):
        raise NotImplementedError

class DEfit(FitBase):
    def solve(self, **kw):
        from mystic.optimizer import de
        from mystic.solver import Minimizer
        strategy = de.DifferentialEvolution(**kw)
        minimize = Minimizer(strategy=strategy, problem=self.problem)
        x = minimize()
        return x

class SNOBfit(FitBase):
    def solve(self, **kw):
        from snobfit.snobfit import snobfit
        self.lasttime = self.time-61
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        x,fx,calls = snobfit(self.problem, self.problem.guess(), bounds,
                             fglob=0, callback=self._monitor)
        return x
    def _monitor(self, k, x, fx, improved):
        t = time.clock()
        if improved and self.lasttime < t-60:
            self.lasttime = t
            parameter.summarize(self.problem.parameters)
        print k,(fx if improved else "")

class Result:
    def __init__(self, problem, solution):
        problem.setp(solution)
        self.problem = problem
        self.parameters = deepcopy(problem.parameters)

    def show(self):
        """
        Show the model parameters and plots
        """
        self.showmodel()
        self.showpars()
        # Show the graph if pylab is available
        try:
            import pylab
        except:
            pass
        else:
            self.problem.plot()
            pylab.show()

    def showmodel(self):
        print "== Model parameters =="
        self.problem.show()

    def showpars(self):
        print "== Fitted parameters =="
        parameter.summarize(self.parameters)

class FitProblem:
    def __init__(self, fitness):
        self.fitness = fitness
        self._prepare()
    def model_parameters(self):
        """
        Parameters associated with the model.
        """
        return self.fitness.parameters()
    def model_points(self):
        """
        Number of data points associated with the model.
        """
        return len(self.fitness.probe.Q)
    def model_update(self):
        """
        Update the model according to the changed parameters.
        """
        self.fitness.update()
    def model_nllf(self):
        """
        Negative log likelihood of seeing data given model.
        """
        return self.fitness.nllf()
    def guess(self):
        """
        Return the user values as the guess for the initial model.
        """
        return [p.value for p in self.parameters]
    def _prepare(self):
        """
        Prepare for the fit.

        This sets the parameters and the bounds properties that the
        solver is expecting from the fittable object.  We also compute
        the degrees of freedom so that we can return a normalized fit
        likelihood.
        """
        all_parameters = parameter.unique(self.model_parameters())
        self.parameters = parameter.varying(all_parameters)
        self.bounded = [p for p in all_parameters
                       if not isinstance(p.bounds, mbounds.Unbounded)]
        self.dof = self.model_points() - len(self.parameters)
        #self.constraints = pars.constraints()
    def setp(self, pvec):
        """
        Set a new value for the parameters into the model.  If the model
        is valid, calls model_update to signal that the model should be
        recalculated.
        
        Returns True if the value is valid and the parameters were set,
        otherwise returns False.
        """
        #TODO: do we have to leave the model in an invalid state?
        # WARNING: don't try to conditionally update the model
        # depending on whether any model parameters have changed.
        # For one thing, the model_update below probably calls
        # the subclass MultiFitProblem.model_update, which signals
        # the individual models.  Furthermore, some parameters may
        # related to others via expressions, and so a dependency
        # tree needs to be generated.  Whether this is better than
        # clicker() from SrFit I do not know.
        for v,p in zip(pvec,self.parameters):
            p.value = v
        valid = all((p.value in p.bounds) for p in self.bounded)
        #self.constraints()
        if valid:
            self.model_update()
        return valid
    def parameter_nllf(self):
        """
        Returns negative log likelihood of seeing parameters p.
        """
        return numpy.sum(p.nllf() for p in self.bounded)
    def parameter_residuals(self):
        """
        Returns negative log likelihood of seeing parameters p.
        """
        return [p.residual() for p in self.bounded]
    def residuals(self):
        """
        Return the model residuals.
        """
        return self.fitness.residuals()
    def chisq(self):
        """
        Return sum squared residuals normalized by the degrees of freedom.

        In the context of a composite fit, the reduced chisq on the individual
        models only considers the points and the fitted parameters within
        the individual model.
        
        Note that this does not include cost factors due to constraints on
        the parameters, such as sample_offset ~ N(0,0.01).
        """
        #model_dof = self.model_points() - len(self.parameters)
        return numpy.sum(self.residuals()**2)/self.dof
    def __call__(self, pvec):
        """
        Compute the cost function for a new parameter set p.

        Note that this is not simply the sum-squared residuals, but instead
        is the negative log likelihood of seeing the data given the model plus
        the negative log likelihood of seeing the model.  The individual 
        likelihoods are scaled by 1/max(P) so that normalization constants 
        can be ignored.  The log likelihood is further scaled by 2/DOF so that
        the result looks like the familiar normalized chi-squared.  These
        scale factors will not affect the value of the minimum, though some
        care will be required when interpreting the uncertainty.
        """
        if self.setp(pvec):
            try:
                if isnan(self.parameter_nllf()):
                    print "Parameter nllf is wrong"
                    for p in self.bounded:
                        print p,p.nllf()
                cost = self.model_nllf() + self.parameter_nllf()
            except KeyboardInterrupt:
                raise
            except:
                #TODO: make sure errors get back to the user
                import traceback
                traceback.print_exc()
                parameter.summarize(self.parameters)
                return inf
            if isnan(cost):
                #TODO: make sure errors get back to the user
                print "point evaluates to NaN"
                parameter.summarize(self.parameters)
                return inf
            # Make cost look like
            return 2*cost/self.dof
        else:
            return inf
    def show(self):
        print parameter.format(self.model_parameters())
        print "[chisq=%g]"%self.chisq()

    def plot(self):
        self.fitness.plot()
        import pylab
        pylab.text(0,0,'chisq=%g'%self.chisq(),
                   transform=pylab.gca().transAxes)

class MultiFitProblem(FitProblem):
    """
    Weighted fits for multiple models.
    """
    def __init__(self, models, weights=None):
        self.fits = [FitProblem(m) for m in models]
        if weights is None:
            weights = [1 for m in models]
        self.weights = weights
        self._prepare()
    def model_parameters(self):
        """Return parameters from all models"""
        return [f.model_parameters() for f in self.fits]
    def model_points(self):
        """Return number of points in all models"""
        return numpy.sum(f.model_points() for f in self.fits)
    def model_update(self):
        """Let all models know they need to be recalculated"""
        # TODO: consider an "on changed" signal for model updates.
        # The update function would be associated with model parameters
        # rather than always recalculating everything.  This
        # allows us to set up fits with 'fast' and 'slow' parameters,
        # where the fit can quickly explore a subspace where the
        # computation is cheap before jumping to a more expensive
        # subspace.  SrFit does this.
        for f in self.fits: f.model_update()
    def model_nllf(self):
        """Return cost function for all data sets"""
        return numpy.sum(f.model_nllf() for f in self.fits)
    def residuals(self):
        resid = numpy.hstack([w*f.residuals()
                              for w,f in zip(self.weights,self.fits)])
        return resid

    def show(self):
        L = []
        for i,f in enumerate(self.fits):
            print "-- Model %d"%i
            f.show()
        print "[overall chisq=%g]"%self.chisq()

    def plot(self):
        import pylab
        for i,f in enumerate(self.fits):
            pylab.figure(i+1)
            f.plot()
