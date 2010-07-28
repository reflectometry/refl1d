import time
import thread
import base64

import dill

from park import jsonrpc

from park import config
from park.jsonrpc import ServerProxy
from park.server import JobService
from park.direct import Scheduler as LocalScheduler

class WithServerProxy(ServerProxy):
    def __enter__(self):
        """
        Set the default service for subsequent requests.

        Example::

            from park.client import *
            with connect('http://parkserver.org'):
                M1 = Assembly((model,data))
                fit = Fit(M1)
                err = Uncertainty(M1, fit.result, after=fit)
            result = err.wait()
        """
        service_stack.append(self)
        return self

    def __exit__(self, *args):
        service_stack.pop()

class WithJobService(JobService):
    def __enter__(self):
        service_stack.append(self)
        return  self
    def __exit__(self):
        service_stack.pop()

def _local_service():
    return WithJobService(scheduler=LocalScheduler())

def connect(server=""):
    if server is "":
        return _local_service()
    else:
        return WithServerProxy(server)

service_stack = [connect(config.jobserver())]
def default_server():
    """
    Return the current default service if no service is specified.

    This is usually local_service, but it can be a remote service
    if the function is called in the context of "with remote:"
    """
    return service_stack[-1]


def decode_kernel(env, input):
    return dill.loads(base64.b64decode(input))
def encode_kernel(kernel):
    return base64.b64encode(dill.dumps(kernel))

class Job(object):
    def __init__(self, requires=[], service=None, kernel=None):
        self.requires = requires
        self.service = service
        if callable(kernel):
            self.kernel = dict(name="park.client.decode_kernel",
                               input=encode_kernel(kernel))
        else:
            self.kernel = kernel
    def submit(self, server=None):
        if server is None:
            server = default_server()
        try:
            job = dict(requires=self.requires,
                       service=self.service,
                       kernel=self.kernel)
            jobid = server.submit(job)
        except jsonrpc.Fault, exc:
            parts = exc.args[1]
            raise RuntimeError("\n".join((parts["message"],parts["traceback"])))
        except:
            raise
        return JobProxy(server,jobid)

class JobProxy(object):
    """
    Proxy for remotely executing job.
    """
    class JobCancelled(Exception): pass
    def __init__(self, server, jobid):
        self.server = server
        self.jobid = jobid
    @property
    def status(self):
        """
        Query job status.
        """
        return self.server.status()

    def wait(self, pollrate=1):
        """
        Wait for job to complete.
        """
        while True:
            status = self.server.status(jobid)
            if status == "DONE":
                return self.server.result(jobid)
            elif status == "ERROR":
                error = self.server.fetch('error')
                raise error
            elif status == "CANCEL":
                raise JobProxy.JobCancelled()
            time.sleep(pollrate)

    def after(self, fn, pollrate=1):
        """
        Function to call after job is complete.
        """
        thread.start_new_thread(self._monitor,(fn, pollrate))
    def _monitor(self, fn, pollrate):
        result = self.wait(pollrate=pollrate)
        fn(result)
        
