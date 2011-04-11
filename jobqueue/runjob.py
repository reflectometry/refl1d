import os
import sys
import traceback

from . import store
from . import services

def build_command(id, request):
    path = store.path(id)
    script = """
import os
from jobqueue import runjob, store
#import sys; print "\\n".join(sys.path)
id = "%s"
request = store.get(id,"request")
runjob.run(id, request)
"""%id
    scriptfile = os.path.join(path,'runner.py'%id)
    open(scriptfile,'w').write(script)
    return sys.executable+" "+scriptfile

def run(id, request):
    #print "\n".join(sys.path)
    path = store.path(id)
    store.create(id)  # Path should already exist, but just in case...
    os.chdir(path)    # Make sure the program starts in the path
    sys.stdout = open(os.path.join(path,'stdout.txt'),'w')
    sys.stderr = open(os.path.join(path,'stderr.txt'),'w')
    service = getattr(services, request['service'], None)
    if service is None:
        result = {
                  'status': 'ERROR',
                  'trace': 'service <%s> not available'%request['service'],
                  }
    else:
        try:
            val = service(request)
        except:  # Capture the service error, whatever it may be
            result = {
                      'status': 'ERROR',
                      'trace': traceback.format_exc(),
                      }
        else:
            result = {
                      'status': 'COMPLETE',
                      'result': val,
                      }

    store.put(id,'results', result)

def results(id):
    results = store.get(id,'results')
    if results is None:
        raise RuntimeError("Results for %d cannot be empty"%id)
    return results