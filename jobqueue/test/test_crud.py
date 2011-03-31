import sys; sys.path.append('../..')

from jobqueue.client import connect

server = connect('http://localhost:5000')

print "list",server.joblist()
job = server.submit(dict(model='reflectivity'))
print "submit",job
print "list",server.joblist()
print "fetch",server.job(job['id'])
print "delete",server.delete(job['id'])
print "list",server.joblist()
