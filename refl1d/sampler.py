"""
MCMC samplers for reflectometry models.
"""
import numpy
import dream
from . import fitter

class Model(dream.MCMCModel):
    """
    DREAM wrapper for refl1d models.
    """
    def __init__(self, model=None):
        """
        Create a sampling from the multidimensional likelihood function
        represented by the problem set using dream.
        """
        self.model = model
        self.bounds = zip(*[p.bounds.limits for p in model.parameters])
        self.labels = [p.name for p in model.parameters]

    def nllf(self, x):
        """Negative log likelihood of seeing models given parameters *x*"""
        return self.model.nllf(x)
    
    def plot(self, x):
        """Display the contents of the model in the current figure"""
        self.model.setp(x)
        self.model.plot()

def draw_samples(models=None, weights=None, chains=10, **kw):
    """
    Draw random samples from the likelihood surface of the models.
    """
    problem = fitter._make_problem(models=models, weights=weights)
    model = Model(model=problem)
    pop_size = chains*len(problem.parameters)
    population = random_population(problem, pop_size)
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
