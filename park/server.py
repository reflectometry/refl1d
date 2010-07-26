import re
import json

from park.jobid import get_jobid
from park.slurm import Scheduler
from park.store import Store
from park import environment

# Keys must not have any characters for changing directories
_keypattern = re.compile('^[-._a-zA-Z0-9]*$')
def _validate_key(s):
    if _keypattern.match(s) is None:
        raise ValueError("key not valid: %s"%s)
def _validate_jobid(s):
    if _keypattern.match(s) is None:
        raise ValueError("jobid not valid: %s"%s)

def _validate_job(job):
    pass

class JobService:
    """
    Set of commands to interact with a job queue and results store.
    """
    def __init__(self, scheduler, store):
        self._scheduler = scheduler
        self._store = store

    def jobs(self):
        """
        Show available jobs.
        """
        return self._scheduler.jobs()

    def fetch(self, jobid, key):
        _validate_jobid(jobid)
        _validate_key(key)
        return self._store.get(jobid, key)

    def stored(self, jobid):
        _validate_jobid(jobid)
        return self._store.keys(jobid)

    def store(self, jobid, key, value):
        _validate_jobid(jobid)
        _validate_jobid(key)
        self._store.put(jobid, key, value)

    def delete(self, jobid):
        """
        Delete a job and all associated data.
        """
        _validate_jobid(jobid)
        self._scheduler.cancel(jobid)
        self._store.destroy(jobid)

    def cancel(self, jobid):
        """
        Cancel a job but don't delete its data.
        """
        _validate_jobid(jobid)
        self._scheduler.cancel(jobid)

    def submit(self, job):
        """
        Submit a job to the queue, returning the job id.
        """
        _validate_job(job)
        jobid = get_jobid(job)
        service,kernel = environment.commands(jobid, job)
        #print "submitting",job,"\nas",jobid,"\nusing",service,"\nand",kernel
        self._scheduler.submit(jobid, service, kernel)
        return jobid

def json_server(service):
    from jsonrpc import SimpleJSONRPCServer as Server
    print 'Running JSON-RPC server on port 8000'
    server = Server(("localhost", 8000))
    server.register_instance(service)
    server.serve_forever()

def start():
    json_server(JobService(scheduler=Scheduler(), store=Store()))
