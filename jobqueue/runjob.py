import os
import sys
import traceback

from . import store

SERVICE = {}

def count(request, path):
    total = 0
    for i in range(request['data']):
        total += 1
    return total
SERVICE['count'] = count

def build_command(id, request):
    path = store.path(id)
    script = """
import os
from jobqueue import runjob, store
#import sys; print "\\n".join(sys.path)
import refl1d.fitservice # Ack! Improper service registration
id = %s
request = store.get(id,"request")
runjob.run(id, request)
"""%id
    scriptfile = os.path.join(path,'J%s.py'%id)
    open(scriptfile,'w').write(script)
    return sys.executable+" "+scriptfile

def run(id, request):
    path = store.path(id)
    sys.stdout = open(os.path.join(path,'stdout.txt'),'w')
    sys.stderr = open(os.path.join(path,'stderr.txt'),'w')
    try:
        val = SERVICE[request['service']](request, path)
        result = {
            'status': 'COMPLETE',
            'result': val,
            }
    except:
        result = {
            'status': 'ERROR',
            'trace': traceback.format_exc(),
          }
    store.put(id,'results', result)

def results(id):
    return store.get(id,'results')