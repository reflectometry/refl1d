from jobqueue import runjob
from jobqueue.client import connect

POLLRATE = 60

def wait_for_result(remote, id, process):
    """
    Wait for job processing to finish.  Meanwhile, prefetch the next
    request.
    """
    next_request = None
    cancelling = False
    while True:
        process.join(POLLRATE)
        if not process.is_alive(): break
        ret = remote.status(id)
        if ret.status == 'CANCEL':
            process.terminate()
            cancelling = True
            break
        if not next_request:
            next_request = remote.nextjob(queue=queue)

    try:
        result = runjob.result(id)
    except KeyError:
        if cancelling:
            result = { 'status': 'CANCEL' }
        else:
            result = { 'status': 'ERROR' }

    return result, next_request

def update_remote(remote, id, queue, result):
    remote.postjob(id, result, queue)
    path= store.path()
    files = [os.path.join(path,f) for f in os.path.dirlist(path)]
    remote.putfiles(id, files, queue)

def serve(dispatcher, queue):

    assert queue is not None
    next_request = None
    remote = connect(dispatcher)
    while True:
        if not next_request:
            next_request = remote.nextjob(queue=queue)
        if next_request:
            jobid = next_request['id']
            store.create(id)
            process = Process(target=runjob.run, args=(jobid,next_request))
            process.start()
            result, next_request = wait_for_result(remote, jobid, process)
            thread.start_thread(update_remote, args=(remote, jobid, queue))
        else:
            time.sleep(POLLRATE)

if __name__ == "__main__":
    serve(dispatcher='http://reflectometry.org:5000',
          queue="sparkle.ncnr.nist.gov")
