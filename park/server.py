# TODO: make a restful http server with binary upload capabilities for large files
# Formats .json, .xml, .csv, .txt or none for .html
# GET jobs{format}        : list of jobs
# GET job/{jobid}/store/{key}   : particular key
# GET job/{jobid}/results
# GET job/{jobid}/store/{key}
# etc.
import re
import json
import base64

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

    def results(self, jobid):
        self._scheduler.deactivate(jobid)
        return self._store.get(jobid,'results')

    def store(self, jobid, key, value):
        """
        Store a value on the server associated with a key (e.g., a filename).

        Value must be base64 encoded.
        """
        _validate_jobid(jobid)
        _validate_jobid(key)
        value = base64.b64decode(value)
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

    def prepare(self, job):
        """
        Prepare the job for running, returning the job id, but do not
        start it.  If a job has associated data.
        """
        _validate_job(job)
        jobid = get_jobid(job)
        self._store.create(jobid)
        self._store.put(jobid, 'job', json.dumps(job))
        return jobid
    
    def start(self, jobid):
        _validate_jobid(jobid)
        job = json.loads(self._store.get(jobid, 'job'))
        self._queue(jobid, job)

    def submit(self, job):
        """
        Submit a job to the queue, returning the job id.
        
        This is equivalent to start(prepare(job))
        """
        jobid = self.prepare(job)
        self._queue(jobid, job)
        return jobid
        
    def _queue(self, jobid, job):
        service,kernel = environment.commands(jobid, job)
        #print "submitting",job,"\nas",jobid,"\nusing",service,"\nand",kernel
        self._scheduler.queue_service(jobid, service, kernel, 
                                      jobdir=self._store.path(jobid))

def json_server(service):
    from jsonrpc import SimpleJSONRPCServer as Server
    print 'Running JSON-RPC server on port 8000'
    server = Server(("localhost", 8000))
    server.register_instance(service)
    server.serve_forever()

def start():
    json_server(JobService(scheduler=Scheduler(), store=Store()))
