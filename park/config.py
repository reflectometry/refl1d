import os
import os.path

def _root():
    return os.path.join(os.environ['HOME'],'.park')

def interp():
    return "python"

def env():
    return {'PYTHONPATH': '/home/pkienzle/danse/refl1d'}

def tasks():
    return 8

def virtualdir():
    return os.path.join(_root(),'env')

def storagedir():
    return os.path.join(_root(),'store')

def jobid_path():
    return os.path.join(_root(),'jobid')

def jobserver():
    ## Client is running on the server machine
    return "http://sparkle.ncnr.nist.gov:8000"
    #return "http://localhost:8000"
    ## Client is running on a machine that can handle local jobs
    #return ""