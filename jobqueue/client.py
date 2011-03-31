import json
from . import restful

json_content = ('Content-Type','application/json')

class Connection(object):
    def __init__(self, url):
        self.rest = restful.Connection(url)
    def joblist(self):
        response = self.rest.request_get('/jobs.json')
        return json.loads(response['body'])
    def submit(self, job):
        data = json.dumps(job)
        response = self.rest.request_post('/jobs.json',
                                          headers=dict([json_content]),
                                          body=data)
        print response['body']
        return json.loads(response['body'])
    def job(self, id):
        response = self.rest.request_get('/jobs/%d.json'%id)
        return json.loads(response['body'])
    def delete(self, id):
        response = self.rest.request_delete('/jobs/%d.json'%id)
        return json.loads(response['body'])

def connect(url):
    return Connection(url)
