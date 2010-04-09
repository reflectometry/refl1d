"""
Adaptors for fitting.

*WARNING* within models self.parameters() returns a tree of all possible
parameters associated with the model.  Within fit problems, self.parameters
is a list of fitted parameters only.
"""
from copy import deepcopy
from numpy import inf, nan, isnan
import numpy
from mystic import parameter, bounds as mbounds, monitor
from mystic.formatnum import format_uncertainty
import time

class FitBase:
    def __init__(self, problem):
        """Fit the models and show the results"""
        self.problem = problem
    def solve(self, **kw):
        raise NotImplementedError

class ConsoleMonitor(monitor.TimedUpdate):
    def __init__(self, problem, progress=1, improvement=30):
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.problem = problem
    def show_progress(self, history):
        print "step",history.step[0],"chisq",history.value[0]
    def show_improvement(self, history):
        #print "step",history.step[0],"chisq",history.value[0]
        self.problem.setp(history.point[0])
        parameter.summarize(self.problem.parameters)

class DEfit(FitBase):
    def solve(self, **kw):
        from mystic.optimizer import de
        from mystic.solver import Minimizer
        monitors = kw.pop('monitors',None)
        if monitors == None:
            monitors = [ConsoleMonitor(self.problem)]
        strategy = de.DifferentialEvolution(**kw)
        minimize = Minimizer(strategy=strategy, problem=self.problem,
                             monitors=monitors)
        x = minimize()
        return x

class AmoebaFit(FitBase):
    def solve(self, **kw):
        from simplex import simplex
        self.best = numpy.inf
        self.lasttime = time.clock()
        self.lastfx = self.best
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        result = simplex(f=self.problem, x0=self.problem.guess(), bounds=bounds,
                         update_handler=self._monitor, **kw)
        return result.x
    def _monitor(self, k, n, x, fx):
        t = time.clock()
        improved = self.best > fx
        if fx < self.lastfx and self.lasttime < t-60:
            self.lasttime = t
            self.lastfx = fx
            parameter.summarize(self.problem.parameters)
            print "step %d of %d"%(k,n),(fx if improved else "")
        return True
        
class SNOBfit(FitBase):
    def solve(self, **kw):
        from snobfit.snobfit import snobfit
        self.lasttime = time.clock()-61
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

def preview(models=[], weights=None):
    """Preview the models in preparation for fitting"""
    problem = _make_problem(models=models, weights=weights)
    result = Result(problem,problem.guess())
    result.show()
    return result

def fit(models=[], weights=None, fitter=DEfit, **kw):
    """
    Perform a fit
    """
    problem = _make_problem(models=models, weights=weights)
    if fitter is not None:
        t0 = time.clock()
        opt = fitter(problem)
        x = opt.solve(**kw)
        print "time",time.clock()-t0
    else:
        x = problem.guess()
    result = Result(problem,x)
    result.show()
    return result


class Result:
    def __init__(self, problem, solution):
        problem.setp(solution)
        self.chisq = [problem.chisq()]
        self.samples = numpy.array([solution])
        self.problem = problem
        self.parameters = deepcopy(problem.parameters)

    def resample(self, samples=100, restart=False, fitter=AmoebaFit, **kw):
        """
        Refit the result multiple times with resynthesized data, building
        up an array in Result.samples which contains the best fit to the
        resynthesized data.  *samples* is the number of samples to generate.  
        *fitter* is the (local) optimizer to use. **kw are the parameters
        for the optimizer.
        """
        opt = fitter(self.problem)
        chisqs = []
        trials = []
        try: # TODO: some solvers already catch KeyboardInterrupt
            for i in range(samples):
                print "== resynth %d of %d"%(i,samples)
                self.problem.resynth_data()
                if restart:
                    parameter.randomize(self.problem.parameters)
                else:
                    self.problem.setp(self.samples[0])
                x = opt.solve(**kw)
                self.problem.setp(x)
                chisq = self.problem.chisq()
                trials.append(x)
                chisqs.append(chisq)
                parameter.summarize(self.problem.parameters)
                print "[chisq=%g]"%chisq
        except KeyboardInterrupt:
            pass
        self.samples = numpy.vstack([self.samples]+trials)
        self.chisq = self.chisq+chisqs

        # Restore the original solution
        self.problem.restore_data()
        self.problem.setp(self.samples[-1])

    def show_stats(self):
        if self.samples.shape[0] > 1:
            chisq = format_uncertainty(numpy.mean(self.chisq),numpy.std(self.chisq,ddof=1))
            lo,hi = min(self.chisq),max(self.chisq)
            print "Chisq for samples: %s in [%g,%g]"%(chisq,lo,hi)
            parameter.show_stats(self.parameters, self.samples)
            parameter.show_correlations(self.parameters, self.samples)

    def save(self, basename):
        """
        Save the parameter table and the fitted model.
        """
        # TODO: need to do problem.setp(solution) in case the problem has
        # changed since result was created (e.g., when comparing multiple
        # fits). Same in showmodel()
        fid = open(basename+".par","w")
        parameter.summarize(self.parameters, fid=fid)
        fid.close()
        self.problem.save(basename)
        if self.samples.shape[0] > 1:
            fid = open(basename+".mc","w")
            fid.write("# "+"\t".join([p.name for p in self.parameters]))
            numpy.savetxt(fid, self.samples.T, delimiter="\t")
            fid.close()
        return self
        
    def show(self):
        """
        Show the model parameters and plots
        """
        self.showmodel()
        self.showpars()
        # Show the graph if pylab is available
        try:
            import pylab
            import atexit
        except:
            pass
        else:
            self.problem.plot()
            atexit.register(pylab.show)
        return self

    def showmodel(self):
        print "== Model parameters =="
        self.problem.show()

    def showpars(self):
        print "== Fitted parameters =="
        parameter.summarize(self.parameters)

def _make_problem(models=[], weights=None):
    if isinstance(models,(tuple,list)):
        problem = MultiFitProblem(models, weights=weights)
    else:
        problem = FitProblem(models)
    return problem


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
    def resynth_data(self):
        """Resynthesize data with noise from the uncertainty estimates."""
        self.fitness.probe.resynth_data()
    def restore_data(self):
        """Restore original data after resynthesis."""
        self.fitness.probe.restore_data()
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

    def save(self, basename):
        self.fitness.save(basename)

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
    def resynth_data(self):
        """Resynthesize data with noise from the uncertainty estimates."""
        for f in self.fits: f.resynth_data()
    def restore_data(self):
        """Restore original data after resynthesis."""
        for f in self.fits: f.restore_data()
    def residuals(self):
        resid = numpy.hstack([w*f.residuals()
                              for w,f in zip(self.weights,self.fits)])
        return resid

    def save(self, basename):
        for i,f in enumerate(self.fits):
            f.save(basename+"-%d"%(i+1))
        
    def show(self):
        for i,f in enumerate(self.fits):
            print "-- Model %d"%i
            f.show()
        print "[overall chisq=%g]"%self.chisq()

    def plot(self):
        import pylab
        for i,f in enumerate(self.fits):
            pylab.figure(i+1)
            f.plot()
