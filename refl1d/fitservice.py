import os
import sys
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')

from . import cli
from . import __version__

# Site configurate determines what kind of mapper to use
# This should be true in cli.py as well
from .mapper import MPMapper as mapper

def fitservice(request):

    path = os.getcwd()

    service_version = __version__
    request_version = str(request['version'])
    if service_version != request_version:
        raise ValueError('fitter version %s does not match request %s'
                         % (service_version, request_version))

    data = request['data']
    model = str(data['package'])
    if model != 'refl1d':
        raise ValueError('model is not refl1d')

    service_model_version = __version__
    request_model_version = str(data['version'])
    if service_model_version != request_model_version:
        raise ValueError('%s version %s does not match request %s'
                         % (model, service_model_version, request_model_version))
    options = pickle.loads(str(data['options']))
    problem = pickle.loads(str(data['problem']))
    problem.store = path
    problem.output_path = os.path.join(path,'model')

    if options.fit == 'dream':
        fitter = cli.DreamProxy(problem=problem, opts=options)
    else:
        fitter = cli.FitProxy(cli.FITTERS[options.fit],
                              problem=problem, options=options)

    fitter.mapper = mapper.start_mapper(problem, options.args)
    problem.show()
    print "#", " ".join(sys.argv)
    best, fbest = fitter.fit()
    cli.remember_best(fitter, problem, best)
    matplotlib.pyplot.show()
    return list(best), fbest
