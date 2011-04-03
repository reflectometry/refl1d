import os
import sys
import shutil
import json
import traceback

ROOT = '/tmp/serve/%s'
SERVICE = {}

def count(request, path):
    total = 0
    for i in range(request['data']):
        total += 1
    return {'total': total}
SERVICE['count'] = count


def run_one(id, request):
    path = ROOT%id
    try: os.makedirs(path)
    except: pass
    sys.stdout = open(os.path.join(path,'stdout'),'w')
    sys.stderr = open(os.path.join(path,'stderr'),'w')
    try:
        result = SERVICE[request['service']](request,path)
        result['status'] = 'COMPLETE'
    except:
        result = {
            'status': 'ERROR',
            'trace': traceback.format_exc(),
          }
    open(os.path.join(path,'result'),'w').write(json.dumps(result))

def fetch_result(id):
    try:
        result = json.loads(open(os.path.join(ROOT%id,'result'),'r').read())
        # Remove top level unicode from json converter
        result = dict((str(k),v) for k,v in result.items())
    except:
        result = {
            'status': 'ERROR',
            'trace': traceback.format_exc(),
            }
    return result

def clean_result(id):
    shutil.rmtree(ROOT%id)
