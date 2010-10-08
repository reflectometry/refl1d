import time
import thread
import base64

from park import json, export
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

# Set up the default server based on config.jobserver().  If the
# default server is "", then a direct server will be used, otherwise
# a remote server will be used.  In the case where the client is
# running on a machine that acts as a remote server, the default
# job server should be set to the server address so that local jobs
# go through the same job queue as remote jobs.
service_stack = [connect(config.jobserver())]
def default_server(server=None):
    """
    Return the current default server if no server is specified.

    This is usually local_service, but it can be a remote service
    if the function is called in the context of "with server:"
    """
    if server is None:
        return service_stack[-1]
    else:
        return server

@export
def user_kernel(env, input):
    if config.allow_user_code():
        return kernel_loads(input)
    else:
        raise TypeError("server does not allow user code to run")
@export
def named_kernel(env, input):
    from park.util import import_symbol
    return import_symbol(input)
def make_kernel(kernel):
    if callable(kernel):
        kernel = dict(name="park.client.user_kernel",
                      input=kernel_dumps(kernel))
    elif not isinstance(kernel,dict):
        kernel = dict(name="park.client.named_kernel",input=kernel)
    return kernel
def kernel_dumps(kernel):
    import dill
    return base64.b64encode(dill.dumps(kernel))
def kernel_loads(input):
    import dill
    return dill.loads(base64.b64decode(input))

class JobDescription(object):
    def __init__(self, requires=[], service=None, kernel=None):
        self.requires = requires
        kernel = make_kernel(kernel)
        files = kernel.pop('files',{})
        files.update(service.pop('files',{}))
        self.service = service
        self.kernel = kernel
        self.files = files
    def submit(self, server=None):
        server = default_server(server)
        job = dict(requires=self.requires,
                   service=self.service,
                   kernel=self.kernel)
        jobid = server.prepare(job)
        for key,filename in self.files.items():
            data = fileload(filename)
            server.store(jobid,key,base64.b64encode(data))
        server.start(jobid)
        return Job(server, jobid, job=job)

def fileload(filename):
    fid = open(filename,'rb')
    data = fid.read()
    fid.close()
    return data

class Job(object):
    """
    Proxy for remotely executing job.
    """
    def __init__(self, server, jobid, job=None):
        self.server = server
        self.jobid = jobid
        if job: self._job = job

    def status(self):
        """
        Query remote server for job status.  Returns one of:

            PENDING, ACTIVE, COMPLETE, ERROR
        """
        return self.server.status(self.jobid)

    @property
    def job(self):
        """
        Query remote server for job description.
        """
        if not hasattr(self, '_job'):
            job = json.loads(self.server.fetch(self.jobid,'job'))
            self._job = JobDescription(**job)
        return self._job

    @property
    def result(self):
        """
        Check if job is complete, and return the value or return None if
        job is not yet complete.

        Raises RuntimeError if the job terminated with an error.

        Example:

            while job.result is None: time.sleep(1)
            print job.result
        """
        if hasattr(self, '_result'):
            return self._result
        elif hasattr(self, '_error'):
            raise RuntimeError("job raised remote error")

        status = self.status()
        if status in ['ACTIVE','PENDING']:
            return None
        elif status == "COMPLETE":
            self._result = self.server.result(self.jobid)
            return self._result
        elif status == "ERROR":
            self._error = self.server.fetch(self.jobid,'error')
            raise RuntimeError("job raised remote error")
        else:
            raise RuntimeError("unknown job status %s"%status)

    @property
    def error(self):
        """
        Return the error if one has been encounter, or return None.

        This does not check for job completion first, and so it should
        be used after job.result or job.wait() raises an error.

        Example:

            try:
                print job.wait()
            except:
                print job.error
        """
        if hasattr(self, '_error'):
            return self._error
        else:
            return None

    def wait(self, pollrate=1):
        """
        Wait for job to complete.
        """
        while self.result is None:
            time.sleep(pollrate)
        return self.result

    def after(self, fn, pollrate=1):
        """
        Asynchronous job completion notification.

        *fn* : f(job, result) -> None
             Function to call when the job is complete.

        If *result* is None then an error occurred during job processing,
        or during communication with the job server.
        """
        thread.start_new_thread(self._after,(fn, pollrate))
    def _after(self, fn, pollrate):
        try:
            fn(self, self.wait(pollrate=pollrate))
        except:
            fn(self, None)

# Debugging helper: fake completed job
class CompletedJob:
    def __init__(self, job, result, error):
        self.job = job
        self.result = result
        self.error = error
    def status(self):
        return "COMPLETE" if self.result else "ERROR"
    def wait(self, pollrate=1):
        if self.result is not None:
            return self.result
        else:
            raise RuntimeError("job raised remote error")
    def after(self, fn, pollrate=1):
        thread.start_new_thread(fn, (self, self.result))
