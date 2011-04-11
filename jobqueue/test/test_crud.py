import sys; sys.path.append('../..')
from jobqueue.client import connect

DEBUG = True

server = connect('http://localhost:5000')

def checkqueue(pending=[], active=[], complete=[]):
    qpending = server.jobs('PENDING')
    qactive = server.jobs('ACTIVE')
    qcomplete = server.jobs('COMPLETE')
    if DEBUG: print "pending",qpending,"active",qactive,"complete",qcomplete
    #assert pending == qpending
    #assert active == qactive
    #assert complete == qcomplete

long = {'service':'count','data':100000,
        'name':'long count','notify':'me'}
short = {'service':'count','data':200,
        'name':'short count','notify':'me'}

job = server.submit(long)
print "submit",job
checkqueue()
job2 = server.submit(short)
print "submit",job2
result = server.wait(job['id'], pollrate=10, timeout=120)
print "result",result
checkqueue()
print "fetch",server.info(job['id'])
print "delete",server.delete(job['id'])
checkqueue()
