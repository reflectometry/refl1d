from __future__ import with_statement
import tempfile
import threading
import thread

class Scheduler(object):
    def __init__(self):
        self._nextid = 0
        self._queue = []
        self._lock = thread.allocate_lock()
        self._block_queue = threading.Condition()
        self._active = None
        thread.start_new_thread(self.run)

    def run(self):
        while True:
            # Process items on the queue
            with self._lock():
                if len(self._queue) > 0:
                    self._active = self._queue.pop(0)
                else:
                    self._active = None
            if self._active:
                self._workers = run_work(self._active)
                continue
            
            # Wait for the next item to be placed on the queue
            self._block_queue.acquire()
            self._block_queue.wait()
            self._block_queue.release()
        
    def jobs(self):
        """
        Return a list of jobs on the queue.
        """
        with self._lock:
            if self._active is not None:
                queue = [self._active] + self._queue
            else:
                queue = self._queue
            return "\n".join("%d %s"%(jobid,job['name']) 
                             for jobid,job in queue)

    # Job specific commands
    def submit(self, job):
        """
        Put a command on batch queue, returning its job id.
        """
        with self._lock:
            self._nextid += 1
            self._queue.append((self._nextid,job))
        # Let queue know there is another item.
        self._block_queue.acquire()
        self._block_queue.notify()
        self._block_queue.release()

    def cancel(self, jobid):
        """
        Remove a possibly running job from the queue.
        """
        with self._lock:
            if self._active is not None and jobid == self._active[0]:
                kill_work(self._workers)
            else:
                self._queue = [item for item in self._queue 
                               if item[0] != jobid]

# An in-memory version of store --- may be a bad idea, particularly for
# jobs which generate large log files.
class Store:
    def __init__(self):
        self._store = {}
        self._work = {}
    def create(self, jobid):
        self._store[jobid] = {}
    def destroy(self, jobid):
        del self._store[jobid]
    def keys(self, jobid):
        return self._store[jobid].keys()
    def put(self, jobid, key, value):
        self._store[jobid][key] = value
    def add(self, jobid, key, value):
        self._store[jobid][key] += value        
    def get(self, jobid, key):
        return self._store[jobid][key]
    def delete(self, jobid, key):
        del self._store[jobid][key]
    def put_workfile(self, jobid, key, value):
        fid = tempfile.NamedTemporaryFile(suffix=key, prefix='park', delete=False)
        if not jobid in self._work:
            self._work[jobid] = {}
        self._work[jobid][key] = fid.name
        fid.write(value)
        fid.close()
    def get_workfile(self, jobid, key):
        return self._work[jobid][key]
