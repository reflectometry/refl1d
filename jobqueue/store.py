import os
import json
import shutil

ROOT = '/tmp/server/%s'

def path(id):
    return ROOT%id

def create(id):
    try: os.makedirs(path(id))
    except: pass

def destroy(id):
    shutil.rmtree(path(id))

def put(id, key, value):
    value = json.dumps(value)
    datapath = path(id)
    datafile = os.path.join(datapath,"J%s-%s.json"%(id,key))
    try:
        open(datafile,'wb').write(value)
    except:
        raise KeyError("Could not store key %s-%s"%(id,key))

def get(id, key):
    datapath = path(id)
    datafile = os.path.join(datapath,"J%s-%s.json"%(id,key))
    try:
        value = open(datafile,'rb').read()
    except:
        raise KeyError("Could not retrieve key %s-%s"%(id,key))
    return json.loads(value)

def contains(id, key):
    datapath = path(id)
    datafile = os.path.join(datapath,"J%s-%s.json"%(id,key))
    return os.path.exists(datafile)

def delete(id, key):
    datapath = path(id)
    datafile = os.path.join(datapath,"J%s-%s.json"%(id,key))
    try:
        os.unlink(datafile)
    except:
        raise KeyError("Could not delete key %s-%s"%(id,key))

