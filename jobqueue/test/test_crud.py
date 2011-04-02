import sys; sys.path.append('../..')

from jobqueue.client import connect

server = connect('http://localhost:5000')

print "list",server.joblist()
job = server.submit({'service':'count','count':1000000})
print "submit",job
print "list",server.joblist()
print "pending",server.joblist(status='PENDING')
print "active",server.joblist(status='ACTIVE')
print "complete",server.joblist(status='COMPLETE')
job2 = server.submit({'service':'count','count':200})
print "submit",job2
result = server.wait(job['id'],pollrate=10)
print "result",result
print "pending",server.joblist(status='PENDING')
print "complete",server.joblist(status='COMPLETE')
print "fetch",server.info(job['id'])
print "delete",server.delete(job['id'])
print "list",server.joblist()
