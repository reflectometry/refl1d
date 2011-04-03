import cPickle as pickle

import jobqueue

from . import cli

# Site configurate determines what kind of mapper to use
# This should be true in cli.py as well
from .mapper import MPMapper as mapper

def fitservice(request, path):

    service_version = refl1d.__version__
    request_version = str(request['version'])
    if service_version != request_version:
        raise ValueError('fitter version %s does not match request %s'
                         % (service_version, request_version))

    data = request['data']
    model = str(data['model'])
    if model != 'refl1d':
        raise ValueError('model is not refl1d')

    service_version = refl1d.__version__
    request_version = str(data['version'])
    if service_version != request_version:
        raise ValueError('%s version %s does not match request %s'
                         % (model, service_version, request_version))
    options = pickle.loads(str(data['options']))
    problem = pickle.loads(str(data['problem'])) 

    problem.store = path
    if options.fit == 'dream':
        fitter = cli.DreamProxy(problem=problem, opts=options)
    else:
        fitter = cli.FitProxy(cli.FITTERS[options.fit], 
                              problem=problem, options=options)

    fitter.mapper = mapper.start_mapper(problem, opts.args)
    problem.output_path = path
    problem.show()
    print "#", " ".join(sys.argv)
    best, fbest = fitter.fit()
    remember_best(fitter, problem, best)
    

jobqueue.SERVICE['fitter'] = fitservice

