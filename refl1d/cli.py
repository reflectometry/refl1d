#!/usr/bin/env python

import sys
import os

import shutil
import subprocess

import numpy
import dream
from refl1d.fitter import DEFit, AmoebaFit, SnobFit, BFGSFit

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
    def __init__(self, problem, pop=10, iters=1000, burn=0):
        self.dream_model = DreamModel(problem)
        self.pop = pop
        self.iters = iters
        self.burn = burn
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
                              draws = pop_size*(self.iters+self.burn),
                              burn = pop_size*self.burn)

        self.state = sampler.sample()
        self.state.title = self.dream_model.problem.name

        return self.state.best()[0]

    def save(self, output):
        self.state.save(output)

    def plot(self, output):
        self.state.show(figfile=output)

class FitProxy(object):
    def __init__(self, fitter, problem):

        self.fitter = fitter
        self.problem = problem
        self.opts = {}
    def fit(self):
        import time
        from refl1d.fitter import Result
        if self.fitter is not None:
            t0 = time.clock()
            opt = self.fitter(self.problem)
            x = opt.solve(**self.opts)
            print "time", time.clock() - t0
        else:
            x = self.problem.guess()

        self.result = Result(self.problem, x)
        return x

    def save(self, output):
        pass
        #self.result.show()

    def plot(self, output):
        pass



# ===== Model manipulation ====

def random_population(problem, pop_size):
    """
    Generate a random population from the problem parameters.
    """
    # Generate a random population
    population = [p.bounds.random(pop_size) for p in problem.parameters]
    population = numpy.array(population).T

    # Plug in the initial guess
    guess = problem.guess()
    if guess != None:
        population[0] = numpy.asarray(guess)

    return population


def load_problem(args):
    #import refl1d.context
    file, options = args[0], args[1:]
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
        problem = ctx["problem"]
    except AttributeError:
        raise ValueError(file+" does not define 'problem=FitProblem(models)'")
    problem.file = file
    problem.options = options
    return problem

def preview(model):
    model.show()
    model.plot()
    import pylab; pylab.show()

def remember_best(fitter, problem, best):
    fitter.save(problem.output)

    try:
        problem.save(problem.output, best)
    except:
        pass
    sys.stdout = open(problem.output+".out","w")

    fitter.plot(problem.output)
    problem.show()

    # Plot
    problem.plot(fignum=6, figfile=problem.output)
    import pylab; pylab.suptitle(":".join((problem.store,problem.title)))

def make_store(problem, opts):
    # Determine if command line override
    if opts.store != None:
        problem.store = opts.store
    problem.output = os.path.join(problem.store,problem.name)

    # Check if already exists
    if not opts.overwrite and os.path.exists(problem.output+'.out'):
        if opts.batch:
            print >>sys.stderr, problem.output+" already exists.  Use --overwrite to replace."
            sys.exit(1)
        print problem.output,"already exists."
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
        sys.stdout = open(problem.output+".mon","w")

    # Show command line arguments and initial model
    print "#"," ".join(sys.argv)
    problem.show()


def run_profile(model):
    from refl1d.util import profile
    p = random_population(model,1000)
    print p.shape
    profile(map,model.nllf,p)

# ==== Mappers ====

class SerialMapper:
    @staticmethod
    def start_worker(model):
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
        if '?' in flags or 'h' in flags or 'help' in flags:
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


class FitOpts(ParseOpts):
    MINARGS = 1
    FLAGS = set(("preview","check","profile",
                 "worker","batch","overwrite","parallel"))
    VALUES = set(("plot","store","fit",
                  "pop","iters","burn"))
    pop="10"
    iters="1000"
    burn="0"
    FITTERS= "de","dream","snobfit","amoeba"
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
    --iters=n (default 1000)
        number of fit iterations
    --burn=n (default 0)
        number of iterations before accumulating stats (dream)
    --parallel
        run fit using all processors

Options can be placed anywhere on the command line.

The modelfile is a Python script (i.e., a series of Python commands)
which sets up the data, the models, and the fittable parameters.
The model arguments are available in the modelfile as sys.argv[1:].
Model arguments may not start with '-'.
"""%{'fitter':'|'.join(FITTERS),
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



# ==== Main ====

def main():
    if len(sys.argv) == 1:
        sys.argv.append("-?")
        print "\nNo modelfile parameter was specified.\n"

    opts = FitOpts(sys.argv)
    opts.iters = int(opts.iters)
    opts.pop = int(opts.pop)
    opts.burn = int(opts.burn)
    
    problem = load_problem(opts.args)
    if opts.fit == 'dream':
        fitter = DreamProxy(problem=problem,
                            pop=opts.pop,
                            iters=opts.iters,
                            burn=opts.burn)
    elif opts.fit == 'de':
        fitter = FitProxy(fitter=DEFit, problem=problem)
    elif opts.fit == 'bfgs':
        fitter = FitProxy(fitter=BFGSFit, problem=problem)
    if opts.parallel or opts.worker:
        mapper = AMQPMapper
    else:
        mapper = SerialMapper

    # Which format to view the plots
    from refl1d import Probe
    Probe.view = opts.plot


    if opts.profile:
        run_profile(problem)
    elif opts.worker:
        mapper.start_worker(problem)
    elif opts.check:
        problem.show()
    elif opts.preview:
        preview(problem)
    else:
        make_store(problem,opts)
        fitter.mapper = mapper.start_mapper(problem, opts.args)
        best = fitter.fit()
        remember_best(fitter, problem, best)
        if not opts.batch:
            import pylab; pylab.show()
