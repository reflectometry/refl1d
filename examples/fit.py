#!/usr/bin/env python

import sys
sys.path.extend(('C:/home/pkienzle/danse/refl1d',
                 'C:/home/pkienzle/danse/refl1d/dream'))
import os
import shutil
import subprocess

import numpy
import dream

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
        
        self.mapper = mapper if mapper else lambda p: map(self.log_density,p)

        # Pull things up from 
        self.store = problem.store
        self.dream_opts = problem.dream_opts
        self.name = problem.name
        self.file = problem.file
        self.options = problem.options
        self.setp = problem.setp
        self.title = problem.title

    def nllf(self, x):
        """Negative log likelihood of seeing models given parameters *x*"""
        #print "eval",x; sys.stdout.flush()
        return self.problem.nllf(x)

    def plot(self, x = None, **kw):
        """Display the contents of the model in the current figure"""
        if x is not None: self.problem.setp(x)
        self.problem.plot(**kw)
        
    def show(self, x = None):
        """Display the contents of the model in the current figure"""
        if x is not None: self.problem.setp(x)
        self.problem.show()
        
    def save(self, output, x = None):
        """Display the contents of the model in the current figure"""
        if x is not None: self.problem.setp(x)
        self.problem.save(output)
        
    def map(self, pop):
        return numpy.array(self.mapper(pop))

def draw_samples(model, **kw):
    """
    Draw random samples from the likelihood surface of the models.
    """
    chains = kw.pop('chains',10)
    pop_size = chains*len(model.problem.parameters)
    population = random_population(model.problem, pop_size)
    sampler = dream.Dream(model=model, population=population, **kw)
    
    state = sampler.sample()
    #dream.plot_state(state)
    return state


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

    # Return the population
    return population[None,:,:]


def load_problem(model_file, model_options):
    ctx = dict(__file__=model_file)
    argv = sys.argv
    sys.argv = [model_file] + model_options
    execfile(model_file, ctx) # 2.x
    #exec(compile(open(model_file).read(), model_file, 'exec'), ctx) # 3.0
    sys.argv = argv
    if "problem" not in ctx:
        raise ValueError(model_file+" does not define <problem>")
    ctx["problem"].file = model_file
    ctx["problem"].options = model_options
    return ctx["problem"]

def preview(model):
    print "show"
    model.show()
    model.plot()
    import pylab; pylab.show()
    
def start_fit(model, fit_options):

    try: os.mkdir(model.store)
    except: pass
    output = os.path.join(model.store,model.name)
    shutil.copy2(model.file, model.store)
    
    # Record call and model definition
    sys.stdout = open(output+".mon","w")
    print "#"," ".join(sys.argv)
    model.show()

    #mapper = None
    state = draw_samples(model=model, **model.dream_opts)

    # Save results
    state.title = model.name
    state.save(output)
    model.save(output, state.best()[0])
    sys.stdout = open(output+".out","w")
    model.show()

    # Plot
    import pylab
    model.plot(fignum=6, figfile=output)
    pylab.suptitle(":".join((model.store,model.title)))
    state.show(figfile=output)
    if not "--noplot" in fit_options: pylab.show()

def start_worker(model):
    #sys.stdout = open("dream-%d.log"%os.getpid(),"w")
    #print "worker is starting"; sys.stdout.flush()
    from amqp_map.config import SERVICE_HOST
    from amqp_map.core import connect, start_worker as serve
    server = connect(SERVICE_HOST)
    #print "worker is serving"; sys.stdout.flush()
    serve(server, "dream", model.log_density)
    #print "worker ended"; sys.stdout.flush()

def start_mapper(model_file, model_options):
    from amqp_map.config import SERVICE_HOST
    from amqp_map.core import connect, Mapper

    server = connect(SERVICE_HOST)
    mapper = Mapper(server, "dream")
    for i in range(8):
        #print "starting",sys.argv[0],"in",os.getcwd()
        cmd = [sys.argv[0], "--worker", model_file] + model_options
        pipe = subprocess.Popen(cmd, universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #os.system(" ".join(cmd+["&"]))
    return mapper

def parse_opts():
    if len(sys.argv) == 1:
        print "Usage: reflfit [options] modelfile [args]"
        sys.exit()
    for i,v in enumerate(sys.argv[1:]):
        if not v.startswith('-'): break
    else:
        print "no model file specified"
        sys.exit()
    fit_options = sys.argv[1:i+1]
    model_file = sys.argv[i+1]
    model_options = sys.argv[i+2:]
    return fit_options, model_file, model_options

def main():
    preview_only = "--preview" in sys.argv
    if preview_only: sys.argv.remove("--preview")        
    fit_options, model_file, model_options = parse_opts()
    problem = load_problem(model_file, model_options)
    model = DreamModel(problem=problem)
    if preview_only:
        preview(model)
    elif "--worker" in fit_options:
        # This is the worker process.
        start_worker(model)
    else:
        # This is the master process.
        #model.mapper = start_mapper(model_file, model_options)
        start_fit(model, fit_options)

if __name__ == "__main__": main()
