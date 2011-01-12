from __future__ import with_statement
__all__ = []

import sys
import os
import shutil
import subprocess

import numpy
import pylab

from refl1d.stajconvert import load_mlayer
from refl1d.fitter import DEFit, AmoebaFit, SnobFit, BFGSFit, RLFit, FitProblem
from refl1d import util
from refl1d.mystic import parameter
from refl1d.probe import Probe

class FitProxy(object):
    def __init__(self, fitter, problem, moniter ,opts):

        self.fitter = fitter
        self.problem = problem
        self.moniter = moniter
        self.opts = opts
       
    def fit(self):
        import time
        from refl1d.fitter import Result
        if self.fitter is not None:
            t0 = time.clock()
            optimizer = self.fitter(self.problem)
            x = optimizer.solve(steps=int(self.opts.steps),
                                monitors = self.moniter,
                                burn=int(self.opts.burn),
                                pop=int(self.opts.pop))
            print "time", time.clock() - t0
        else:
            x = self.problem.getp()

        self.result = x
        self.problem.setp(x)
        return x

    def show(self):
        self.problem.show()

    def save(self, output_path):
        pass

    def plot(self, output_path):
        P = self.problem
        pylab.suptitle(":".join((P.store,P.title)))
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

def random_population(problem, pop_size):
    """
    Generate a random population from the problem parameters.
    """
    # Generate a random population
    population = [p.bounds.random(pop_size) for p in problem.parameters]
    population = numpy.array(population).T

    # Plug in the initial guess
    guess = problem.getp()
    if guess != None:
        population[0] = numpy.asarray(guess)

    return population


def load_staj(file):
    M = load_mlayer(file)
    
    # Exclude unlikely fitting parameters
    exclude = set((M.sample[0].thickness,
               M.sample[-1].thickness,
               M.sample[-1].interface,
               M.probe.back_absorption,
               ))
    if M.probe.intensity.value == 1:
        exclude.add(M.probe.intensity)
    if M.probe.background.value < 2e-10:
        exclude.add(M.probe.background.value)
    ## Zero values are excluded below
    #if M.probe.theta_offset.value == 0:
    #    exclude.add(M.probe.theta_offset)
    #for L in M.sample:
    #    if L.rho.value == 0: exclude.add(L.rho)
    #    if L.irho.value == 0: exclude.add(L.irho)
    
    # Fit everything else using a range of +/- 20 %
    for p in parameter.unique(M.parameters()):
        if p in exclude: continue
        if p.value != 0: p.pmp(20)
        #p.fixed = False

    job = FitProblem(M)
    job.file = file
    job.title = file
    job.name = file
    job.options = []
    return job

def load_job(args):
    #import refl1d.context
    file, options = args[0], args[1:]
    
    if file.endswith('.staj'):
        return load_staj(file)
    
    ctx = dict(__file__=file)
    #refl1d.context.math_context(ctx)
    #refl1d.context.refl_context(ctx)
    #refl1d.context.fitting_context(ctx)
    argv = sys.argv
    sys.argv = [file] + options
    execfile(file, ctx) # 2.x
    #exec(compile(open(model_file).read(), model_file, 'exec'), ctx) # 3.0
    sys.argv = argv
    try:
        job = ctx["problem"]
    except AttributeError:
        raise ValueError(file+" does not define 'problem=FitProblem(models)'")
    job.file = file
    job.options = options
    return job

# ==== Mappers ====

class SerialMapper:
    @staticmethod
    def start_worker(model):
        pass
    @staticmethod
    def start_mapper(job, modelargs):
        return lambda points: map(job.nllf, points)
    @staticmethod
    def stop_mapper(mapper):
        pass


# ==== option parser ====

class ParseOpts:
    MINARGS = 0
    FLAGS = set()
    VALUES = set()
    USAGE = ""
    def __init__(self, args):
        self._parse(args)

    def _parse(self, args):
        
        flagargs = [v for v in sys.argv[0:] if v.startswith('-') and not '=' in v]
        flags = set(v[0:] for v in flagargs)
        if '?' in flags or 'h' in flags or 'help' in flags:
            print self.USAGE
            sys.exit()
        unknown = flags - self.FLAGS
        if any(unknown):
            raise ValueError("Unknown options -%s.  Use -? for help."%", -".join(unknown))
        for f in self.FLAGS:
            setattr(self, f, (f in flags))

        valueargs = [v for v in sys.argv[0:] if v.startswith('-') and '=' in v]
        for f in valueargs:
            idx = f.find('=')
            name = f[0:idx]
            value = f[idx+0:]
            if name not in self.VALUES:
                raise ValueError("Unknown option -%s. Use -? for help."%name)
            setattr(self, name, value)

        positionargs = [v for v in sys.argv[0:] if not v.startswith('-')]
        self.args = positionargs
        


FITTERS = dict(dream=None, rl=RLFit,
               de=DEFit, newton=BFGSFit, amoeba=AmoebaFit, snobfit=SnobFit)
class FitOpts(ParseOpts):
    MINARGS = 1
    FLAGS = set(("preview","check","profile","edit",
                 "worker","batch","overwrite","parallel"))
    VALUES = set(("plot","store","mesh","meshsteps",
                  "fit","pop","steps","burn",
                 ))
    pop="10"
    steps="100"
    burn="0"
    PLOTTERS="log","linear","fresnel","q4"
    USAGE = """\
Usage: reflfit [-option] modelfile [modelargs]

where options include:

    -?/-h/--help
        display this help
    --check
        print the model description and chisq value and exit
    --preview
        display model but do not perform a fitting operation
    --batch
        batch mode; don't show plots during run
    --plot=%(plotter)s
        type of reflectivity plot to display
    --store=path
        output directory for plots and models
    --overwrite
        if store already exists, replace it
    --fit=%(fitter)s (default de)
        fitting engine to use; see manual for details
    --pop=n (default 10)
        population size per parameter (used for dream and DE)
    --steps=n (default 1000)
        number of fit iterations
    --burn=n (default 0)
        number of iterations before accumulating stats (dream)
    --parallel
        run fit using all processors
    --mesh=var OR var+var
        plot chisq line or plane
    --meshsteps=n
        number of steps in the mesh

For mesh plots, var can be a fitting parameter with optional
range specifier, such as:

   P[0].range(3,6)

or the complete path to a model parameter:

   M[0].sample[1].material.rho.pm(1)

Options can be placed anywhere on the command line.

The modelfile is a Python script (i.e., a series of Python commands)
which sets up the data, the models, and the fittable parameters.
The model arguments are available in the modelfile as sys.argv[1:].
Model arguments may not start with '-'.
"""%{'fitter':'|'.join(sorted(FITTERS.keys())),
     'plotter':'|'.join(PLOTTERS),
     }

    _plot = 'log'
    def _set_plot(self, value):
        if value not in set(self.PLOTTERS):
            raise ValueError("unknown plot type %s; use %s"%(value,"|".join(self.PLOTTERS)))
        self._plot = value
    plot = property(fget=lambda self: self._plot, fset=_set_plot)
    store = None
    _fitter = 'de'
    def _set_fitter(self, value):
        if value not in set(self.FITTERS):
            raise ValueError("unknown fitter %s; use %s"%(value,"|".join(self.FITTERS)))
        self._fitter = value
    fit = property(fget=lambda self: self._fitter, fset=_set_fitter)
    meshsteps = 40

