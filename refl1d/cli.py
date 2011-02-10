from __future__ import with_statement

import sys
import os

import shutil
import subprocess

import numpy
import pylab
import dream
from .stajconvert import load_mlayer, fit_all
from .fitter import DEFit, AmoebaFit, SnobFit, BFGSFit, RLFit, PTFit
from . import fitter
from . import util
from .mystic import parameter
from .probe import Probe

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
        self.steps = opts.steps
        self.burn = opts.burn
    def _get_mapper(self):
        return self.dream_model.mapper
    def _set_mapper(self, mapper):
        self.dream_model.mapper = mapper
    mapper = property(fget=_get_mapper, fset=_set_mapper)

    def fit(self):
        pop_size = self.pop*len(self.dream_model.problem.parameters)
        population = random_population(self.dream_model.problem, pop_size)
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
    def __init__(self, fitter, problem, opts):
        self.fitter = fitter
        self.problem = problem
        self.opts = opts
    def fit(self):
        import time
        from .fitter import Result
        if self.fitter is not None:
            t0 = time.clock()
            optimizer = self.fitter(self.problem)
            x = optimizer.solve(steps=self.opts.steps,
                                burn=self.opts.burn,
                                pop=self.opts.pop,
                                CR=self.opts.CR,
                                Tmin=self.opts.Tmin,
                                Tmax=self.opts.Tmax,
                                )
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

def load_problem(args):
    filename = args[0]
    if filename.endswith('.staj'):
        model = load_mlayer(filename, fit_pmp=20)
        options = []
        problem = fitter.FitProblem(model)
    else:
        options = args[1:]
        problem = fitter.load_problem(filename, options=options)

    problem.file = filename
    problem.title = os.path.basename(filename)
    problem.name = filename
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

    # Plot

def make_store(problem, opts):
    # Determine if command line override
    if opts.store != None:
        problem.store = opts.store
    problem.output_path = os.path.join(problem.store,problem.name)

    # Check if already exists
    if not opts.overwrite and os.path.exists(problem.output_path+'.out'):
        if opts.batch:
            print >>sys.stderr, problem.output_path+" already exists.  Use --overwrite to replace."
            sys.exit(1)
        print problem.output_path,"already exists."
        print "Press 'y' to overwrite, or 'n' to abort and restart with --store=newpath"
        ans = raw_input("Overwrite [y/n]? ")
        if ans not in ("y","Y","yes"):
            sys.exit(1)

    # Create it and copy model
    try: os.mkdir(problem.store)
    except: pass
    shutil.copy2(problem.file, problem.store)

    # Redirect sys.stdout to capture progress
    if opts.batch:
        sys.stdout = open(problem.output_path+".mon","w")

    # Show command line arguments and initial model
    print "#"," ".join(sys.argv)
    problem.show()


def run_profile(problem):
    from .util import profile
    p = random_population(problem,1000)
    profile(map,problem.nllf,p)

# ==== Mappers ====

class SerialMapper:
    @staticmethod
    def start_worker(problem):
        pass
    @staticmethod
    def start_mapper(problem, modelargs):
        return lambda points: map(problem.nllf, points)
    @staticmethod
    def stop_mapper(mapper):
        pass

class AMQPMapper:

    @staticmethod
    def start_worker(problem):
        #sys.stderr = open("dream-%d.log"%os.getpid(),"w")
        #print >>sys.stderr,"worker is starting"; sys.stdout.flush()
        from amqp_map.config import SERVICE_HOST
        from amqp_map.core import connect, start_worker as serve
        server = connect(SERVICE_HOST)
        #os.system("echo 'serving' > /tmp/map.%d"%(os.getpid()))
        #print "worker is serving"; sys.stdout.flush()
        serve(server, "dream", problem.nllf)
        #print >>sys.stderr,"worker ended"; sys.stdout.flush()

    @staticmethod
    def start_mapper(problem, modelargs):
        import multiprocessing
        from amqp_map.config import SERVICE_HOST
        from amqp_map.core import connect, Mapper

        server = connect(SERVICE_HOST)
        mapper = Mapper(server, "dream")
        cpus = multiprocessing.cpu_count()
        pipes = []
        for _ in range(cpus):
            cmd = [sys.argv[0], "--worker"] + modelargs
            #print "starting",sys.argv[0],"in",os.getcwd(),"with",cmd
            pipe = subprocess.Popen(cmd, universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pipes.append(pipe)
        for pipe in pipes:
            if pipe.poll() > 0:
                raise RuntimeError("subprocess returned %d\nout: %s\nerr: %s"
                                   % (pipe.returncode, pipe.stdout, pipe.stderr))
        #os.system(" ".join(cmd+["&"]))
        import atexit
        def exit_fun():
            for p in pipes: p.terminate()
        atexit.register(exit_fun)

        #print "returning mapper",mapper
        return mapper

    @staticmethod
    def stop_mapper(mapper):
        for pipe in mapper.pipes:
            pipe.terminate()


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
        if len(positionargs) < self.MINARGS:
            raise ValueError("Not enough arguments. Use -? for help.")
        self.args = positionargs

        #print "flags",flags
        #print "vals",valueargs
        #print "args",positionargs


FITTERS = dict(dream=None, rl=RLFit, pt=PTFit,
               de=DEFit, newton=BFGSFit, amoeba=AmoebaFit, snobfit=SnobFit)
class FitOpts(ParseOpts):
    MINARGS = 1
    FLAGS = set(("preview","check","profile","edit","random","simulate",
                 "worker","batch","overwrite","parallel",
                 ))
    VALUES = set(("plot","store","fit","noise",
                  "CR","Tmin","Tmax","burn","steps","pop",
                  #"mesh","meshsteps",
                 ))
    noise="5"
    CR="0.9"
    Tmin="0.1"
    Tmax="10"
    pop="10"
    steps="1000"
    burn="0"
    PLOTTERS="log","linear","fresnel","q4"
    USAGE = """\
Usage: refl1d modelfile [modelargs] [options]

The modelfile is a Python script (i.e., a series of Python commands)
which sets up the data, the models, and the fittable parameters.
The model arguments are available in the modelfile as sys.argv[1:].
Model arguments may not start with '-'.

Options:

    --preview
        display model but do not perform a fitting operation
    --plot=log      [%(plotter)s]
        type of reflectivity plot to display
    --random
        use a random initial configuration
    --simulate
        simulate the data to fit
    --noise=5%%
        percent noise to add to the simulated data

    --store=path
        output directory for plots and models
    --overwrite
        if store already exists, replace it
    --parallel
        run fit using all processors
    --batch
        batch mode; don't show plots after fit

    --fit=de        [%(fitter)s]
        fitting engine to use; see manual for details
    --steps=1000    [all optimizers]
        number of fit iterations after any burn-in time
    --pop=10        [dream, de, rl, pt]
        population size
    --burn=0        [dream, pt]
        number of burn-in iterations before accumulating stats
    --Tmin=0.1
    --Tmax=10       [pt]
        temperature range; use a higher maximum temperature and a larger
        population if your fit is getting stuck in local minima.
    --CR=0.9        [de, rl, pt]
        crossover ratio for population mixing

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


# ==== Main ====

def main():
    if len(sys.argv) == 1:
        sys.argv.append("-?")
        print "\nNo modelfile parameter was specified.\n"

    opts = FitOpts(sys.argv)
    opts.steps = int(opts.steps)
    opts.pop = float(opts.pop)
    opts.burn = int(opts.burn)
    opts.CR = float(opts.CR)
    opts.Tmin = float(opts.Tmin)
    opts.Tmax = float(opts.Tmax)

    problem = load_problem(opts.args)
    if opts.random:
        problem.randomize()
    if opts.simulate:
        problem.simulate_data(noise=float(opts.noise))
        # If fitting, then generate a random starting point different
        # from the simulation
        if not (opts.edit or opts.check or opts.preview):
            problem.randomize()

    if opts.fit == 'dream':
        fitter = DreamProxy(problem=problem, opts=opts)
    else:
        fitter = FitProxy(FITTERS[opts.fit],
                          problem=problem, opts=opts)
    if opts.parallel or opts.worker:
        mapper = AMQPMapper
    else:
        mapper = SerialMapper

    # Which format to view the plots
    Probe.view = opts.plot


    if opts.edit:
        from .profileview.demo import demo
        demo(problem.fitness)
    elif opts.profile:
        run_profile(problem)
    elif opts.worker:
        mapper.start_worker(problem)
    elif opts.check:
        print "chisq",problem()
    elif opts.preview:
        preview(problem)
    else:
        make_store(problem,opts)
        fitter.mapper = mapper.start_mapper(problem, opts.args)
        best = fitter.fit()
        remember_best(fitter, problem, best)
        if not opts.batch:
            pylab.show()
