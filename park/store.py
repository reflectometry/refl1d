import re
import os.path
import shutil
import glob

from park import config

# TODO: can users store arbitrary length stuff here?

# Keys must not have any characters for changing directories
keymatch = re.compile('^[-._a-zA-Z0-9]*$')

class Store:
    def _dir(self, jobid):
        return os.path.join(config.storagedir(),jobid)
    def _file(self, jobid, key):
        return os.path.join(self._dir(jobid),key)
    def _workfile(self, jobid, key):
        return os.path.join(self._dir(jobid),'work',key)
    path = _dir

    def create(self, jobid):
        path = self._dir(jobid)
        if not os.path.exists(path): os.makedirs(path)

    def destroy(self, jobid):
        path = self._dir(jobid)
        shutil.rmtree(path)

    def keys(self, jobid):
        return [os.path.basename(p) 
                for p in glob.glob(os.path.join(self._dir(jobid),'*'))]

    def delete(self, jobid, key):
        path = self._file(jobid, key)
        os.unlink(path)

    def put(self, jobid, key, value):
        # Make sure we are storing a string
        try:
            value[0]+' '
        except:
            raise ValueError("value is not a string")
        
        # Store the value in a file
        path = self._file(jobid, key)
        fid = open(path,'wb')
        fid.write(value)
        fid.close()

    def add(self, jobid, key, value):
        # Make sure we are storing a string
        try:
            value[0]+' '
        except:
            raise ValueError("value is not a string")
        
        # Store the value in a file
        path = self._file(jobid, key)
        fid = open(path,'ab')
        fid.write(value)
        fid.close()

    def get(self, jobid, key):
        path = self._file(jobid, key)
        if not os.path.exists(path):
            raise ValueError("key not stored: %s"%key)
        fid = open(path,'rb')
        value = fid.read()
        fid.close()
        return value

    def put_workfile(self, jobid, key, value):
        # Store the value in a file
        path = self._workfile(jobid, key)
        if not os.path.exists(os.path.dirname(path)): 
            os.mkdir(os.path.dirname(path))
        fid = open(path, 'wb')
        fid.write(value)
        fid.close()

    def get_workfile(self, jobid, key):
        return self._workfile(jobid, key)
