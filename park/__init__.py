# TODO: cancel job if connection is lost but user hasn't detached client
# this may require server.keep_alive(jobid, 60*n) every n minutes
try:
    import json
except:
    import simplejson as json

def export(fn):
    fn.park_import = True
    return fn
