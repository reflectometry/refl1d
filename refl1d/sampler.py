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
    def __init__(self, model=None, bounds_handling='reflect'):
        """
        Create a sampling from the multidimensional likelihood function
        represented by the problem set using dream.
        """
        self.model = model
        low, high = zip(*[p.bounds.limits for p in model.parameters])
        self.bounds = dream.set_bounds(low, high, style=bounds_handling)

    def nllf(self, x, RNG=numpy.random):
        """Negative log likelihood of seeing models given parameters *x*"""
        return self.model.nllf(x)

def draw_samples(models=None, weights=None, chains=4, **kw):
    """
    Draw random samples from the likelihood surface of the models.
    """
    problem = fitter._make_problem(models=models, weights=weights)
    model = Model(model=problem, bounds_handling='reflect')
    pop_size = chains*len(problem.parameters)
    population = random_population(problem, pop_size)
    sampler = dream.Dream(model=model, population=population, **kw)
    
    state = sampler.sample()
    dream.plot_state(state)
    return state


def random_population(problem, pop_size):
    """
    Generate a random population from the problem parameters.
    """
    # Generate a random population
    ndim = len(problem.parameters)
    pop_size = int(pop_size * ndim)
    population = [p.bounds.random(pop_size) for p in problem.parameters]
    population = numpy.array(population).T

    # Plug in the initial guess
    guess = problem.guess()
    if guess != None:
        population[0] = numpy.asarray(guess)

    # Return the population
    return population
