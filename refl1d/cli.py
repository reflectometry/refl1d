from __future__ import with_statement

from math import ceil

import sys
import os

import shutil
import subprocess

import numpy
import pylab
import dream
from .stajconvert import load_mlayer, fit_all
from .fitter import (DEFit, AmoebaFit, SnobFit, BFGSFit,
                     PSFit, RLFit, PTFit, MultiStart)
from .fitter import StepMonitor, ConsoleMonitor
from .mapper import MPMapper, AMQPMapper, SerialMapper
from . import fitter
from . import util
from .mystic import parameter
from .probe import Probe
from . import garefl
from . import initpop

# ==== Fitters ====

class DreamModel(dream.MCMCModel):
    """
    DREAM wrapper for refl1d models.
    """
    def __init__(self, problem=None, mapper=None):
        """
        Create a sampling from the multidimensional likelihood function
        represented by the problem set using dream.
        """
        self.problem = problem
        self.bounds = zip(*[p.bounds.limits for p in problem.parameters])
        self.labels = [p.name for p in problem.parameters]

        self.mapper = mapper if mapper else lambda p: map(self.nllf,p)

    def nllf(self, x):
        """Negative log likelihood of seeing models given parameters *x*"""
        #print "eval",x; sys.stdout.flush()
        return self.problem.nllf(x)

    def map(self, pop):
        #print "calling mapper",self.mapper
        return -numpy.array(self.mapper(pop))


class DreamProxy(object):
    def __init__(self, problem, opts):
        self.dream_model = DreamModel(problem)
        self.pop = opts.pop
        self.pop_init = opts.init
        self.steps = opts.steps
        self.burn = opts.burn
    def _get_mapper(self):
        return self.dream_model.mapper
    def _set_mapper(self, mapper):
        self.dream_model.mapper = mapper
    mapper = property(fget=_get_mapper, fset=_set_mapper)

    def fit(self):
        pars = self.dream_model.problem.parameters
        pop_size = int(ceil(self.pop*len(pars)))
        if self.pop_init == 'random':
            population = initpop.random(N=pop_size,
                                        pars=pars, include_current=True)
        elif self.pop_init == 'cov':
            cov = self.dream_model.problem.cov()
            population = initpop.cov(N=pop_size,
                                     pars=pars, include_current=False, cov=cov)
        elif self.pop_init == 'lhs':
            population = initpop.cov(N=pop_size,
                                     pars=pars, include_current=True)
        else:
            raise ValueError("Unknown population initializer '%s'"%self.pop_init)
        population = population[None,:,:]
        sampler = dream.Dream(model=self.dream_model, population=population,
                              draws = pop_size*(self.steps+self.burn),
                              burn = pop_size*self.burn)

        self.state = sampler.sample()
        self.state.title = self.dream_model.problem.name

        best = self.state.best()[0]
        self.dream_model.problem.setp(best)
        return best

    def save(self, output_path):
        self.state.save(output_path)

    def show(self):
        self.dream_model.problem.show()

    def plot(self, output_path):
        self.state.show(figfile=output_path)
        P = self.dream_model.problem
        pylab.figure(6)
        pylab.suptitle(":".join((P.store,P.title)))
        P.plot(figfile=output_path)

class FitProxy(object):
    def __init__(self, fitter, problem, options, monitors=None):
        self.fitter = fitter
        self.problem = problem
        self.options = options
        self.monitors = monitors
        self.mapper = lambda p: map(problem.nllf,p)

    def fit(self):
        import time
        if self.fitter is not None:
            t0 = time.clock()
            optimizer = self.fitter(self.problem)
            if self.options.starts > 1:
                optimizer = MultiStart(optimizer)
            x, fx = optimizer.solve(steps=self.options.steps,
                                    burn=self.options.burn,
                                    pop=self.options.pop,
                                    CR=self.options.CR,
                                    Tmin=self.options.Tmin,
                                    Tmax=self.options.Tmax,
                                    monitors=self.monitors,
                                    starts=self.options.starts,
                                    mapper=self.mapper,
                                    )
            print "time", time.clock() - t0
        else:
            x = self.problem.getp()
            fx = inf

        self.result = x
        self.problem.setp(x)
        return x, fx

    def show(self):
        self.problem.show()

    def save(self, output_path):
        pass

    def plot(self, output_path):
        P = self.problem
        pylab.suptitle(": ".join((P.store,P.title)))
        P.plot(figfile=output_path)


def mesh(problem, vars=None, n=40):
    x,y = [numpy.linspace(p.bounds.limits[0],p.bounds.limits[1],n) for p in vars]
    p1, p2 = vars
    def fn(xi,yi):
        p1.value, p2.value = xi,yi
        problem.model_update()
        #parameter.summarize(problem.parameters)
        return problem.chisq()
    z = [[fn(xi,yi) for xi in x] for yi in y]
    return x,y,numpy.asarray(z)

# ===== Model manipulation ====

def load_problem(args):
    filename, options = args[0], args[1:]

    if (filename.endswith('.so') or filename.endswith('.dll')
        or filename.endswith('.dyld')):
        options = []
        problem = garefl.load(filename)
    elif filename.endswith('.staj'):
        options = []
        problem = FitProblem(load_mlayer(filename))
    else:
        options = args[1:]
        problem = fitter.load_problem(filename, options=options)

    problem.file = filename
    problem.title = os.path.basename(filename)
    problem.name, _ = os.path.splitext(os.path.basename(filename))
    problem.options = options
    return problem

def preview(problem):
    problem.show()
    problem.plot()
    pylab.show()

def remember_best(fitter, problem, best):
    fitter.save(problem.output_path)

    #try:
    #    problem.save(problem.output_path, best)
    #except:
    #    pass
    with util.redirect_console(problem.output_path+".out"):
        fitter.show()
    fitter.show()
    fitter.plot(problem.output_path)

    pardata = "".join("%s %.15g\n"%(p.name, p.value)
                      for p in problem.parameters)
    open(problem.output_path+".par",'wt').write(pardata)

def recall_best(problem, path):
    data = open(path,'rt').readlines()
    for par,line in zip(problem.parameters, data):
        par.value = float(line.split()[-1])

def store_overwrite_query_gui(path):
    import wx
    msg_dlg = wx.MessageDialog(None,path+" Already exists. Press 'yes' to overwrite, or 'No' to abort and restart with newpath",'Overwrite Directory',wx.YES_NO | wx.ICON_QUESTION)
    retCode = msg_dlg.ShowModal()
    msg_dlg.Destroy()
    if retCode != wx.ID_YES:
        raise RuntimeError("Could not create path")

def store_overwrite_query(path):
    print path,"already exists."
    print "Press 'y' to overwrite, or 'n' to abort and restart with --store=newpath"
    ans = raw_input("Overwrite [y/n]? ")
    if ans not in ("y","Y","yes"):
        sys.exit(1)

def make_store(problem, opts, exists_handler):
    # Determine if command line override
    if opts.store != None:
        problem.store = opts.store
    problem.output_path = os.path.join(problem.store,problem.name)

    # Check if already exists
    if not opts.overwrite and os.path.exists(problem.output_path+'.out'):
        if opts.batch:
            print >>sys.stderr, path+" already exists.  Use --overwrite to replace."
            sys.exit(1)
        exists_handler(problem.output_path)

    # Create it and copy model
    try: os.mkdir(problem.store)
    except: pass
    shutil.copy2(problem.file, problem.store)

    # Redirect sys.stdout to capture progress
    if opts.batch:
        sys.stdout = open(problem.output_path+".mon","w")


def run_profile(problem, steps):
    """
    Model execution time profiler.

    Run the program with "--profile --steps=N" to generate a function
    profile chart breaking down the cost of evaluating N models.

    Here is the findings from one profiling session::

       23 ms total
        6 ms rendering model
        8 ms abeles
        4 ms convolution
        1 ms setting parameters and computing nllf

    Using the GPU for abeles/convolution will only give us 2-3x speedup.
    """
    from .util import profile
    p = initpop.random(N=steps, pars=problem.parameters)

    # The cost of
    # To get good information from the profiler, you wil
    # Modify this function to obtain different information

    # For gathering stats on just the rendering.
    fits = getattr(problem,'fits',[problem])
    def rendering(p):
        problem.setp(p)
        for f in fits:
            f.fitness._render_slabs()

    #profile(map,rendering,p)
    profile(map,problem.nllf,p)
    #map(problem.nllf,p)

# ==== option parser ====

class ParseOpts:
    MINARGS = 0
    FLAGS = set()
    VALUES = set()
    USAGE = ""
    def __init__(self, args):
        self._parse(args)

    def _parse(self, args):
        flagargs = [v for v in sys.argv[1:] if v.startswith('--') and not '=' in v]
        flags = set(v[2:] for v in flagargs)
        if 'help' in flags or '-h' in sys.argv[1:] or '-?' in sys.argv[1:]:
            print self.USAGE
            sys.exit()
        unknown = flags - self.FLAGS
        if any(unknown):
            raise ValueError("Unknown options --%s.  Use -? for help."%", --".join(unknown))
        for f in self.FLAGS:
            setattr(self, f, (f in flags))

        valueargs = [v for v in sys.argv[1:] if v.startswith('--') and '=' in v]
        for f in valueargs:
            idx = f.find('=')
            name = f[2:idx]
            value = f[idx+1:]
            if name not in self.VALUES:
                raise ValueError("Unknown option --%s. Use -? for help."%name)
            setattr(self, name, value)

        positionargs = [v for v in sys.argv[1:] if not v.startswith('-')]
        self.args = positionargs



FITTERS = dict(dream=None, rl=RLFit, pt=PTFit, ps=PSFit,
               de=DEFit, newton=BFGSFit, amoeba=AmoebaFit,
               snobfit=SnobFit)
class FitOpts(ParseOpts):
    MINARGS = 1
    FLAGS = set(("preview", "check", "profile", "random", "simulate",
                 "worker", "batch", "overwrite", "parallel", "stepmon",
                 "cov"
               ))
    VALUES = set(("plot", "store", "fit", "noise", "steps", "pop",
                  "CR", "burn", "Tmin", "Tmax", "starts", "seed", "init",
                  "pars", "resynth", "transport"
                  #"mesh","meshsteps",
                ))
    pars=None
    resynth="0"
    noise="5"
    starts="1"
    steps="1000"
    pop="10"
    CR="0.9"
    burn="0"
    Tmin="0.1"
    Tmax="10"
    seed=""
    init="lhs"
    PLOTTERS="fresnel", "linear", "log", "q4"
    USAGE = """\
Usage: refl1d modelfile [modelargs] [options]

The modelfile is a Python script (i.e., a series of Python commands)
which sets up the data, the models, and the fittable parameters.
The model arguments are available in the modelfile as sys.argv[1:].
Model arguments may not start with '-'.

Options:

    --preview
        display model but do not perform a fitting operation
    --pars=filename
        initial parameter values; fit results are saved as <modelname>.par
    --plot=log      [%(plotter)s]
        type of reflectivity plot to display
    --random
        use a random initial configuration
    --simulate
        simulate the data to fit
    --noise=5%%
        percent noise to add to the simulated data
    --cov
        compute the covariance matrix for the model when done


    --store=path
        output directory for plots and models
    --overwrite
        if store already exists, replace it
    --parallel
        run fit using all processors
    --transport='amqp|mp|mpi'
        use amqp/multiprocessing/mpi for parallel evaluation
    --batch
        batch mode; don't show plots after fit
    --stepmon
        show details for each step

    --resynth=0
        run resynthesis error analysis for n generations

    --fit=de        [%(fitter)s]
        fitting engine to use; see manual for details
    --steps=1000    [all optimizers]
        number of fit iterations after any burn-in time
    --pop=10        [dream, de, rl, pt, ps]
        population size
    --burn=0        [dream, pt]
        number of burn-in iterations before accumulating stats
    --Tmin=0.1
    --Tmax=10       [pt]
        temperature range; use a higher maximum temperature and a larger
        population if your fit is getting stuck in local minima.
    --CR=0.9        [de, rl, pt]
        crossover ratio for population mixing
    --starts=1      [%(fitter)s]
        number of times to run the fit from random starting points
    --init='lhs|cov|random' [dream]
        population initialization method, with 'lhs' for latin hypersquares,
        'cov' for covariance, and 'random' for uniform within parameter
        distribution.

    --check
        print the model description and chisq value and exit
    -?/-h/--help
        display this help
"""%{'fitter':'|'.join(sorted(FITTERS.keys())),
     'plotter':'|'.join(PLOTTERS),
     }

#    --mesh=var OR var+var
#        plot chisq line or plane
#    --meshsteps=n
#        number of steps in the mesh
#For mesh plots, var can be a fitting parameter with optional
#range specifier, such as:
#
#   P[0].range(3,6)
#
#or the complete path to a model parameter:
#
#   M[0].sample[1].material.rho.pm(1)

    _plot = 'log'
    def _set_plot(self, value):
        if value not in set(self.PLOTTERS):
            raise ValueError("unknown plot type %s; use %s"
                             %(value,"|".join(self.PLOTTERS)))
        self._plot = value
    plot = property(fget=lambda self: self._plot, fset=_set_plot)
    store = None
    _fitter = 'de'
    def _set_fitter(self, value):
        if value not in set(self.FITTERS):
            raise ValueError("unknown fitter %s; use %s"
                             %(value,"|".join(self.FITTERS)))
        self._fitter = value
    fit = property(fget=lambda self: self._fitter, fset=_set_fitter)
    TRANSPORTS = 'amqp','mp','mpi'
    _transport = 'mp'
    def _set_transport(self, value):
        if value not in self.TRANSPORTS:
            raise ValueError("unknown transport %s; use %s"
                             %(value,"|".join(self.TRANSPORTS)))
        self._transport = value
    transport = property(fget=lambda self: self._transport, fset=_set_transport)
    meshsteps = 40

def getopts():
    opts = FitOpts(sys.argv)
    opts.resynth = int(opts.resynth)
    opts.steps = int(opts.steps)
    opts.pop = float(opts.pop)
    opts.CR = float(opts.CR)
    opts.burn = int(opts.burn)
    opts.Tmin = float(opts.Tmin)
    opts.Tmax = float(opts.Tmax)
    opts.starts = int(opts.starts)
    opts.seed = int(opts.seed) if opts.seed != "" else None
    return opts

# ==== Main ====

def main():
    if len(sys.argv) == 1:
        sys.argv.append("-?")
        print "\nNo modelfile parameter was specified.\n"

    opts = getopts()
    if opts.seed is not None:
        numpy.random.seed(opts.seed)

    problem = load_problem(opts.args)
    if opts.pars is not None:
        recall_best(problem, opts.pars)
    if opts.random:
        problem.randomize()
    if opts.simulate:
        problem.simulate_data(noise=float(opts.noise))
        # If fitting, then generate a random starting point different
        # from the simulation
        if not (opts.check or opts.preview):
            problem.randomize()

    if opts.fit == 'dream':
        fitter = DreamProxy(problem=problem, opts=opts)
    else:
        fitter = FitProxy(FITTERS[opts.fit], problem=problem, options=opts)
    if opts.parallel or opts.worker:
        if opts.transport == 'amqp':
            mapper = AMQPMapper
        elif opts.transport == 'mp':
            mapper = MPMapper
        elif opts.transport == 'mpi':
            raise NotImplementedError("mpi transport not implemented")
    else:
        mapper = SerialMapper

    # Which format to view the plots
    Probe.view = opts.plot

    if opts.profile:
        run_profile(problem, steps=opts.steps)
    elif opts.worker:
        mapper.start_worker(problem)
    elif opts.check:
        if opts.cov: print problem.cov()
        print "chisq",problem()
    elif opts.preview:
        if opts.cov: print problem.cov()
        preview(problem)
    elif opts.resynth > 0:
        make_store(problem,opts,exists_handler=store_overwrite_query)
        fid = open(problem.output_path+".rsy",'at')
        fitter.mapper = mapper.start_mapper(problem, opts.args)
        for i in range(opts.resynth):
            problem.resynth_data()
            best, fbest = fitter.fit()
            print "found %g"%fbest
            fid.write('%.15g '%fbest)
            fid.write(' '.join('%.15g'%v for v in best))
            fid.write('\n')
        problem.restore_data()
        fid.close()

    else:
        make_store(problem,opts,exists_handler=store_overwrite_query)

        # Show command line arguments and initial model
        print "#"," ".join(sys.argv)
        problem.show()
        if opts.stepmon:
            fid = open(problem.output_path+'.log', 'w')
            fitter.monitors = [ConsoleMonitor(problem),
                               StepMonitor(fid,fields=['step','value'])]

        fitter.mapper = mapper.start_mapper(problem, opts.args)
        best, fbest = fitter.fit()
        remember_best(fitter, problem, best)
        if opts.cov: print cov(problem)
        if not opts.batch: pylab.show()

