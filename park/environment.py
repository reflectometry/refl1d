import os
import json

from park.util import import_symbol
from park import config

# TODO: create virtual environments on the fly
# that way each package version doesn't have to have its own virtual
# environment kicking around.
# TODO: let user manage virtual environments
# TODO: interpreter requirement specified as e.g., "python>=2.4,<3.0"
# TODO: build python if necessary

class Environment:
    def __init__(self, mapper=None, store=None):
        self.mapper = mapper
        self.store = store

    def run_service(self, jobid, module, input):
        service = import_symbol(module)
        result = service(self, json.loads(input))
        self.mapper.close()
        self.store.put(jobid, 'result', json.dumps(result))

    def get_kernel(self, module, input):
        kernel = import_symbol(module)
        return kernel(self, json.loads(input))

def prepare(job):
    # TODO: define python interperter, etc. from job description
    pass

def commands(jobid, job):
    interp = config.interp()
    service = "%s -m park.amqp.start_service %s %s '%s'"\
        %(interp, jobid, job['service']['name'], 
          json.dumps(job['service']['input']))
    worker = "%s -m park.amqp.start_kernel %s %s '%s'"\
        %(interp, jobid, job['kernel']['name'], 
          json.dumps(job['kernel']['input']))

    return service, worker


_ = '''
def makeenv(requirements):
    """
    Create an execution environment out of the required packages.
    
    Environment is a dictionary of interpreter, package and requires.
    
    E.g.,
    
        requirements = {
          'interpreter': "python2.5",
          'repositories':
          'packages': ["refl1d==0.2"
                       "matplotlib>0.92",
                       "numpy>1.0,<1.2"],
        }
        bin = makeenv(requirements)
    """
    python = requirements["interpreter"]
    package = requirements["package"]
    interpdir = os.path.join(config.virtualdir(),python)
    packagedir = os.path.join(interpdir,package)
    interp = _interp(requirements)
    if not os.path.exists(packagedir):
        # Construct virtual environment
        if not os.path.exists(interpdir): os.makedirs(interpdir)
        os.system(requirements.python + " -m virtualenv " + packagedir)
        # Install required packages
        for p in requirements["packages"]:
            os.system(interp + ' -m easy_install "' + p + '"')
        os.system(interp)

def _interp(requirements):
    python = requirements["interpreter"]
    package = requirements["package"]
    interpdir = os.path.join(config.virtualdir,python)
    packagedir = os.path.join(interpdir,package)
    return os.path.join(packagedir,"bin","python")

def getenv(job, available_workers):
    service = job['service']
    worker = job['worker']
    env = dict(service_interp=_interp(service),
               worker_interp=_interp(worker),
               store=store,
               num_workers=min(available_workers, service['max_workers']))
    return env
'''
