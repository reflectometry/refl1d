import time
import json
from . import restful

json_content = ('Content-Type','application/json')

class Connection(object):
    def __init__(self, url):
        self.rest = restful.Connection(url)

    def jobs(self, status=None):
        """
        List jobs on the server according to status.
        """
        if status is None:
            response = self.rest.request_get('/jobs.json')
        else:
            response = self.rest.request_get('/jobs/%s.json'%status.lower())
        return _process_response(response)['jobs']

    def submit(self, job):
        """
        Submit a job to the server.
        """
        data = json.dumps(job)
        response = self.rest.request_post('/jobs.json',
                                          headers=dict([json_content]),
                                          body=data)
        return _process_response(response)

    def info(self, id):
        """
        Return the job structure associated with id.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.request_get('/jobs/%s.json'%id)
        return _process_response(response)

    def status(self, id):
        """
        Return the job structure associated with id.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.request_get('/jobs/%s/status.json'%id)
        return _process_response(response)

    def output(self, id):
        """
        Return the result from processing the job.

        Raises ValueError if job not found.
        Raises IOError if communication error.

        Check response['status'] for 'COMPLETE','CANCEL','ERROR', etc.
        """
        response = self.rest.request_get('/jobs/%s/results.json'%id)
        return _process_response(response)

    def wait(self, id, pollrate=300, timeout=60*60*24):
        """
        Wait for job to complete, returning output.

        *pollrate* is the number of seconds to sleep between checks
        *timeout* is the maximum number of seconds to wait

        Raises IOError if the timeout is exceeded.
        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        start = time.clock()
        while True:
            results = self.output(id)
            print "waiting: result is",results
            if results['status'] in ('PENDING', 'ACTIVE'):
                #print "waiting for job %s"%id
                if time.clock() - start > timeout:
                    raise IOError('job %s is still pending'%id)
                time.sleep(pollrate)
            else:
                #print "status for %s is"%id,results['status'],'- wait complete'
                return results

    def stop(self, id):
        """
        Stop the job.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.request_post('/jobs/%s?action=stop'%id)
        return _process_response(response)

    def delete(self, id):
        """
        Delete the job and all associated files.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.request_delete('/jobs/%s.json'%id)
        return _process_response(response)

    def nextjob(self, queue):
        # TODO: sign request
        data = json.dumps({'queue': queue})
        response = self.rest.request_post('/jobs/nextjob.json',
                                          headers=dict([json_content]),
                                          body=data)
        return _process_response(response)

    def postjob(self, id, queue, results):
        # TODO: sign request
        data = json.dumps({'queue': queue, 'results': results})
        response = self.rest.request_post('/jobs/%s/postjob'%id,
                                          headers=dict([json_content]),
                                          body=data)
        return _process_response(response)

    def putfiles(self, id, queue, files):
        # TODO: sign request
        data = json.dumps({'queue': queue})
        response = self.rest.request_post('/jobs/%s/data/'%id,
                                          headers=dict([json_content]),
                                          body=data,
                                          files=files)

def _process_response(response):
    #print "response",response
    if response['headers']['status'] == '200':
        return json.loads(response['body'])
    else:
        print response['body']
        raise IOError("server response code %s"%response['headers']['status'])


def connect(url):
    return Connection(url)
