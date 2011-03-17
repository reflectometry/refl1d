"""
Adaptors for fitting.
*WARNING* within models self.parameters() returns a tree of all possible
parameters associated with the model.  Within fit problems, self.parameters
is a list of fitted parameters only.
"""
from __future__ import division, with_statement
import os
import sys
import time
from copy import deepcopy

from numpy import inf, nan, isnan
import numpy

from .mystic import parameter, bounds as mbounds, monitor
from .mystic.formatnum import format_uncertainty
from .mystic.history import History

from .util import pushdir

class ConsoleMonitor(monitor.TimedUpdate):
    def __init__(self, problem, progress=1, improvement=30):
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.problem = deepcopy(problem)
    # TimedUpdate profiles
    def show_progress(self, history):
        print "step", history.step[0], "cost", history.value[0]
    def show_improvement(self, history):
        #print "step",history.step[0],"chisq",history.value[0]
        self.problem.setp(history.point[0])
        print parameter.summarize(self.problem.parameters)
        try:
            import pylab
            pylab.hold(False)
            self.problem.plot()
            pylab.gcf().canvas.draw()
        except:
            raise

class MonitorRunner(object):
    """
    Adaptor which allows non-mystic solvers to accept progress monitors.
    """
    def __init__(self, monitors, problem):
        if monitors == None:
            monitors = [ConsoleMonitor(problem)]
        self.monitors = monitors
        self.history = History(time=1,step=1,point=1,value=1)
        for M in self.monitors:
            M.config_history(self.history)
        self._start = time.time()
    def __call__(self, step, point, value):
        self.history.update(time=time.time()-self._start,
                            step=step, point=point, value=value)
        for M in self.monitors:
            M(self.history)

class FitBase(object):
    def __init__(self, problem):
        """Fit the models and show the results"""
        self.problem = problem
    def solve(self, **kw):
        raise NotImplementedError

class DEFit(FitBase):
    def solve(self, pop=10, steps=100, **kw):
        from mystic.optimizer import de
        from mystic.solver import Minimizer
        from mystic.stop import Steps
        monitor = kw.pop('monitor', None)
        if monitor == None:
            monitor = [ConsoleMonitor(self.problem)]
        else:
            monitor = [monitor]
        strategy = de.DifferentialEvolution(npop=pop)
        minimize = Minimizer(strategy=strategy, problem=self.problem,
                             monitors=monitor,
                             failure=Steps(steps))
        x = minimize()
        return x


class BFGSFit(FitBase):
    def solve(self, steps=2000, **kw):
        from quasinewton import quasinewton, STATUS
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=kw.pop('monitors', None))
        result = quasinewton(self.problem,
                             x0=self.problem.getp(),
                             monitor = self._monitor,
                             itnlimit = steps,
                             )
        code = result['status']
        print "%d: %s" % (code, STATUS[code])
        return result['x']
    def _monitor(self, step, x, fx):
        self._update(step=step, point=x, value=fx)
        return True

class RLFit(FitBase):
    def solve(self, pop=1, steps=2000, burn=10, CR=0.9, **kw):
        from random_lines import random_lines
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=kw.pop('monitors', None))
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        cfo = dict(cost=self.problem,
                   n = len(bounds[0]),
                   x1 = bounds[0],
                   x2 = bounds[1],
                   monitor = self._monitor)
        NP = int(cfo['n']*pop)
        f_best_overall = inf
        if burn < 1: burn = 1
        assert burn<100
        for _ in range(burn):
            result = random_lines(cfo, NP, maxiter=steps, CR=CR)
            satisfied_sc, n_feval, i_best, f_best, x_best = result
            if f_best < f_best_overall:
                f_best_overall = f_best
                x_best_overall = x_best
        # Note: success if satisfied_sc is 4, failure otherwise.
        return x_best_overall
    def _monitor(self, step, x, fx):
        self._update(step=step, point=x, value=fx)
        return True


class PTFit(FitBase):
    def solve(self, pop=10, steps=2000, burn=1000,
              Tmin=0.1, Tmax=10, CR=0.9, **kw):
        from partemp import parallel_tempering
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=kw.pop('monitors', None))
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        T = numpy.logspace(numpy.log10(Tmin),numpy.log10(Tmax),pop)
        history = parallel_tempering(nllf=self.problem.nllf,
                                    p=self.problem.getp(),
                                    bounds=bounds,
                                    #logfile="partemp.dat",
                                    T=T,
                                    CR=CR,
                                    steps=steps,
                                    burn=burn,
                                    monitor=self._monitor)
        return history.best_point
    def _monitor(self, step, x, fx):
        self._update(step=step, point=x, value=2*fx/self.problem.dof)
        return True



class AmoebaFit(FitBase):
    def solve(self, steps=2000, **kw):
        from simplex import simplex
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=kw.pop('monitors', None))
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        result = simplex(f=self.problem, x0=self.problem.getp(), bounds=bounds,
                         update_handler=self._monitor, maxiter=steps)
        return result.x
    def _monitor(self, k, n, x, fx):
        self._update(step=k, point=x, value=fx)

class SnobFit(FitBase):
    def solve(self, **kw):
        from snobfit.snobfit import snobfit
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=kw.pop('monitors', None))
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        x, _, _ = snobfit(self.problem, self.problem.getp(), bounds,
                          fglob=0, callback=self._monitor)
        return x
    def _monitor(self, k, x, fx, improved):
        self._update(step=k, point=x, value=fx)


def preview(models=[], weights=None):
    """Preview the models in preparation for fitting"""
    problem = _make_problem(models=models, weights=weights)
    result = Result(problem, problem.getp())
    result.show()
    return result

def mesh(models=[], weights=None, vars=None, n=40):
    problem = _make_problem(models=models, weights=weights)

    print "initial chisq",problem.chisq()
    x,y = [numpy.linspace(p.bounds.limits[0],p.bounds.limits[1],n) for p in vars]
    p1, p2 = vars
    def fn(xi,yi):
        p1.value, p2.value = xi,yi
        problem.model_update()
        #print parameter.summarize(problem.parameters)
        return problem.chisq()
    z = [[fn(xi,yi) for xi in x] for yi in y]
    return x,y,numpy.asarray(z)


def fit(models=[], weights=None, fitter=DEFit, **kw):
    """
    Perform a fit
    """
    problem = _make_problem(models=models, weights=weights)
    if fitter is not None:
        t0 = time.clock()
        opt = fitter(problem)
        x = opt.solve(**kw)
        print "time", time.clock() - t0
    else:
        x = problem.getp()
    result = Result(problem, x)
    result.show()

    return result

def show_chisq(chisq, fid=None):
    """
    Show chisq statistics on a drawing from the likelihood function.

    dof is the number of degrees of freedom, required for showing the
    normalized chisq.
    """
    if fid is None: fid = sys.stdout
    v,dv = numpy.mean(chisq), numpy.std(chisq, ddof=1)
    lo, hi = min(chisq), max(chisq)

    valstr = format_uncertainty(v, dv)
    print >>fid, "Chisq for samples: %s,  [min,max] = [%g,%g]" % (valstr,lo,hi)

def show_stats(pars, points, fid=None):
    """
    Print a stylized list of parameter names and values with range bars.

    Report mean +/- std of the samples as the parameter values.
    """
    if fid is None: fid = sys.stdout

    val,err = numpy.mean(points, axis=0), numpy.std(points, axis=0, ddof=1)
    data = [(p.name, p.bounds, v, dv) for p,v,dv in zip(pars,val,err)]
    for name,bounds,v,dv in sorted(data, cmp=lambda x,y: cmp(x[0],y[0])):
        position = int(bounds.get01(v)*9.999999999)
        bar = ['.']*10
        if position < 0: bar[0] = '<'
        elif position > 9: bar[9] = '>'
        else: bar[position] = '|'
        bar = "".join(bar)
        valstr = format_uncertainty(v,dv)
        print >>fid, ("%40s %s %-15s in %s"%(name,bar,valstr,bounds))

def show_correlations(pars, points, fid=None):
    """
    List correlations between parameters in descending order.
    """
    if 1: # Use correlation coefficient
        R = numpy.corrcoef(points.T)
        corr = [(i,j,R[i,j])
                for i in range(len(pars))
                for j in range(i+1, len(pars))]
        # Trim those which are not significant
        corr = [(i,j,r) for i,j,r in corr if abs(r) > 0.2]
        corr = list(sorted(corr, cmp=lambda x,y: cmp(abs(y[2]),abs(x[2]))))

    else: # Use ??
        z = util.zscore(points, axis=0)
        # Compute all cross correlations
        corr = [(i,j,xcorr(z[i],z[j]))
                for i in range(j+1, len(pars))
                for j in range(len(pars))]
        # Trim those which are not significant
        corr = [(i,j,r) for i,j,r in corr if abs(r-2) > 0.5]
        # Sort the remaining list
        corr = list(sorted(corr, cmp=lambda x,y: cmp(abs(y[2]-2),abs(x[2]-2))))

    # Print the remaining correlations
    if len(corr) > 0:
        print >>fid, "== Parameter correlations =="
        for i,j,r in corr:
            print >>fid, pars[i].name, "X", pars[j].name, ":", r



import pytwalk
class TWalk:
    def __init__(self, problem):
        self.twalk = pytwalk.pytwalk(n=len(problem.parameters),
                                     U=problem.nllf,
                                     Supp=problem.valid)
    def run(self, N, x0, x1):
        self.twalk.Run(T=N, x0=x0, xp0=x1)
        return numpy.roll(self.twalk.Output, 1, axis=1)

class Result:
    def __init__(self, problem, solution):
        nllf = problem.nllf(solution) # TODO: Shouldn't have to recalculate!
        self.problem = problem
        self.solution = numpy.array(solution)
        self.points = numpy.array([numpy.hstack((nllf,solution))], 'd')

    def mcmc(self, samples=1e5, burnin=None, walker=TWalk):
        """
        Markov Chain Monte Carlo resampler.
        """
        if burnin is None: burnin = int(samples / 10)
        if burnin >= samples: raise ValueError("burnin must be smaller than samples")

        opt = walker(self.problem)
        x0 = numpy.array(self.solution)
        parameter.randomize(self.problem.parameters)
        x1 = self.problem.getp()
        points = opt.run(N=samples, x0=x0, x1=x1)
        self.points = numpy.vstack((self.points, points[burnin:]))
        self.problem.setp(self.solution)

    def resample(self, samples=100, restart=False, fitter=AmoebaFit, **kw):
        """
        Refit the result multiple times with resynthesized data, building
        up an array in Result.samples which contains the best fit to the
        resynthesized data.  *samples* is the number of samples to generate.
        *fitter* is the (local) optimizer to use. **kw are the parameters
        for the optimizer.
        """
        opt = fitter(self.problem)
        points = []
        try: # TODO: some solvers already catch KeyboardInterrupt
            for i in range(samples):
                print "== resynth %d of %d" % (i, samples)
                self.problem.resynth_data()
                if restart:
                    parameter.randomize(self.problem.parameters)
                else:
                    self.problem.setp(self.solution)
                x = opt.solve(**kw)
                nllf = self.problem.nllf(x) # TODO: don't recalculate!
                points.append(numpy.hstack((nllf,x)))
                print parameter.summarize(self.problem.parameters)
                print "[chisq=%g]" % (nllf*2/self.problem.dof)
        except KeyboardInterrupt:
            pass
        self.points = numpy.vstack([self.points] + points)

        # Restore the original solution
        self.problem.restore_data()
        self.problem.setp(self.solution)

    def show_stats(self):
        if self.points.shape[0] > 1:
            self.problem.setp(self.solution)
            show_chisq(self.points[:,0]*2/self.problem.dof)
            show_stats(self.problem.parameters, self.points[:,1:])
            show_correlations(self.problem.parameters, self.points[:,1:])

    def save(self, basename):
        """
        Save the parameter table and the fitted model.
        """
        # TODO: need to do problem.setp(solution) in case the problem has
        # changed since result was created (e.g., when comparing multiple
        # fits). Same in showmodel()
        self.problem.setp(self.solution)
        fid = open(basename + ".par", "w")
        print >>fid, parameter.summarize(self.problem.parameters)
        fid.close()
        self.problem.save(basename)
        if self.points.shape[0] > 1:
            fid = open(basename + ".mc", "w")
            parhead = "\t".join(p.name for p in self.problem.parameters)
            fid.write("# nllf\t%s\n"%parhead)
            numpy.savetxt(fid, self.points, delimiter="\t")
            fid.close()
        return self

    def show(self):
        """
        Show the model parameters and plots
        """
        self.showmodel()
        self.showpars()
        return self

    def plot(self):
        self.problem.plot()
        return self

    def showmodel(self):
        print "== Model parameters =="
        self.problem.setp(self.solution)
        self.problem.show()

    def showpars(self):
        print "== Fitted parameters =="
        self.problem.setp(self.solution)
        print parameter.summarize(self.problem.parameters)


def _make_problem(models=[], weights=None):
    if isinstance(models, (tuple, list)):
        if len(models) > 1:
            problem = MultiFitProblem(models, weights=weights)
        else:
            problem = FitProblem(models[0])
    else:
        problem = FitProblem(models)
    return problem


# Abstract base class
class Fitness(object):
    def parameters(self):
        """
        Return the set of parameters in the model.
        """
        raise NotImplementedError
    def update(self):
        """
        Parameters have been updated; model needs to be re-evaluated.
        """
        raise NotImplementedError
    def numpoints(self):
        """
        Return the number of data points.
        """
        raise NotImplementedError
    def nllf(self):
        """
        Return the negative log likelihood value of the current parameter set.
        """
        raise NotImplementedError
    def resynth_data(self):
        """
        Generate fake data based on uncertainties in the real data.
        """
        raise NotImplementedError
    def restore_data(self):
        """
        Restore the original data in the model.
        """
        raise NotImplementedError
    def residiuals(self):
        """
        Return residuals for current theory minus data.
        """
        raise NotImplementedError
    def save(self, basename):
        """
        Save the model to a file based on basename+extension
        """
        pass

    def plot(self):
        """
        Plot the model to the current figure.
        """
        pass

class FitProblem(object):
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
        return self.fitness.numpoints()
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

    def simulate_data(self, noise=None):
        """Simulate data with added noise"""
        self.fitness.simulate_data(noise=noise)
    def resynth_data(self):
        """Resynthesize data with noise from the uncertainty estimates."""
        self.fitness.resynth_data()
    def restore_data(self):
        """Restore original data after resynthesis."""
        self.fitness.restore_data()
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
    def valid(self, pvec):
        return all(v in p.bounds for p,v in zip(self.parameters,pvec))

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
        for v, p in zip(pvec, self.parameters):
            p.value = v
        #self.constraints()
        self.model_update()
    def getp(self):
        """
        Returns the current value of the parameter vector.
        """
        return numpy.array([p.value for p in self.parameters], 'd')

    def randomize(self):
        """
        Generates a random model.
        """
        for p in self.parameters:
            p.value = p.bounds.random(1)[0]

    def parameter_nllf(self):
        """
        Returns negative log likelihood of seeing parameters p.
        """
        s = numpy.sum(p.nllf() for p in self.bounded)
        #print "\n".join("%s %g"%(p,p.nllf()) for p in self.bounded)
        return s

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
        return numpy.sum(self.residuals()**2) / self.dof
    def nllf(self, pvec=None):
        """
        Compute the cost function for a new parameter set p.

        Note that this is not simply the sum-squared residuals, but instead
        is the negative log likelihood of seeing the data given the model plus
        the negative log likelihood of seeing the model.  The individual
        likelihoods are scaled by 1/max(P) so that normalization constants
        can be ignored.
        """
        if pvec is not None:
            if self.valid(pvec):
                self.setp(pvec)
            else:
                return inf

        try:
            if isnan(self.parameter_nllf()):
                print "Parameter nllf is wrong"
                for p in self.bounded:
                    print p, p.nllf()
            cost = self.model_nllf() + self.parameter_nllf()
        except KeyboardInterrupt:
            raise
        except:
            #TODO: make sure errors get back to the user
            import traceback
            traceback.print_exc()
            print parameter.summarize(self.parameters)
            return inf
        if isnan(cost):
            #TODO: make sure errors get back to the user
            print "point evaluates to NaN"
            print parameter.summarize(self.parameters)
            return inf
        return cost

    def __call__(self, pvec=None):
        """
        Problem cost function.

        Returns the negative log likelihood scaled by DOF so that
        the result looks like the familiar normalized chi-squared.  These
        scale factors will not affect the value of the minimum, though some
        care will be required when interpreting the uncertainty.
        """
        return 2*self.nllf(pvec)/self.dof

    def show(self):
        print parameter.format(self.model_parameters())
        print "[chisq=%g, nllf=%g]" % (self.chisq(), self.nllf()/self.dof)
        print parameter.summarize(self.parameters)

    def save(self, basename):
        self.fitness.save(basename)

    def plot(self, p=None, fignum=None, figfile=None):
        import pylab
        if fignum != None: pylab.figure(fignum)
        if p != None: self.setp(p)
        self.fitness.plot()
        pylab.text(0, 0, 'chisq=%g' % self.chisq(),
                   transform=pylab.gca().transAxes)
        if figfile != None:
            pylab.savefig(figfile+"-model", format='png')

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
    def simulate_data(self, noise=None):
        """Simulate data with added noise"""
        for f in self.fits: f.simulate_data(noise=noise)
    def resynth_data(self):
        """Resynthesize data with noise from the uncertainty estimates."""
        for f in self.fits: f.resynth_data()
    def restore_data(self):
        """Restore original data after resynthesis."""
        for f in self.fits: f.restore_data()
    def residuals(self):
        resid = numpy.hstack([w * f.residuals()
                              for w, f in zip(self.weights, self.fits)])
        return resid

    def save(self, basename):
        for i, f in enumerate(self.fits):
            f.save(basename + "-%d" % (i + 1))

    def show(self):
        for i, f in enumerate(self.fits):
            print "-- Model %d" % i
            f.show()
        print "[overall chisq=%g, nllf=%g]" % (self.chisq(), self.nllf()/self.dof)

    def plot(self,fignum=1,figfile=None):
        import pylab
        for i, f in enumerate(self.fits):
            f.plot(fignum=i+fignum)
            pylab.suptitle('Model %d'%i)
            if figfile != None: pylab.savefig(figfile+"-model%d"%i)

def load_problem(file, options=[]):
    """
    Load a problem definition from a python script file.

    sys.argv is set to ``[file] + options`` within the context of the script.

    The user must define ``problem=FitProblem(...)`` within the script.

    Raises ValueError if the script does not define problem.
    """
    dir,file = os.path.split(file)
    with pushdir(dir):
        ctx = dict(__file__=file)
        argv = sys.argv
        sys.argv = [file] + options
        execfile(file, ctx) # 2.x
        #exec(compile(open(model_file).read(), model_file, 'exec'), ctx) # 3.0
        sys.argv = argv
        try:
            problem = ctx["problem"]
        except AttributeError:
            raise ValueError(file+" does not define 'problem=FitProblem(...)'")

    return problem
