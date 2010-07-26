import sys

from park import config
from park.jsonrpc import ServerProxy
from park.server import JobService
from park.direct import Scheduler as LocalScheduler

class WithServerProxy(ServerProxy):
    def __enter__(self):
        """
        Set the default service for subsequent requests.

        Example::

            from park.client import *
            with connect('http://parkserver.org'):
                M1 = Assembly((model,data))
                fit = Fit(M1)
                err = Uncertainty(M1, fit.result, after=fit)
            result = err.wait()
        """
        service_stack.append(self)
        return self

    def __leave__(self):
        service_stack.pop()

class WithJobService(JobService):
    def __enter__(self):
        service_stack.append(self)
        return  self
    def __leave__(self):
        service_stack.pop()

def _local_service():
    return WithJobService(scheduler=LocalScheduler())

def connect(server=""):
    if server is "":
        return _local_service()
    else:
        return WithServerProxy(server)

service_stack = [connect(config.jobserver())]
def default_server():
    """
    Return the current default service if no service is specified.

    This is usually local_service, but it can be a remote service
    if the function is called in the context of "with remote:"
    """
    return service_stack[-1]


def main():
    server = default_server()
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == 'jobs':
            print server.jobs()
            i += 1
        elif sys.argv[i] == 'cancel':
            server.cancel(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == 'stored':
            print server.stored(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == 'store':
            fid = open(sys.argv[i+3])
            data = fid.read()
            fid.close()
            server.fetch(sys.argv[i+1],sys.argv[i+2],data)
            i += 4
        elif sys.argv[i] == 'fetch':
            data = server.fetch(sys.argv[i+1],sys.argv[i+2])
            print data
            i += 3
        else:
            raise ValueError("unknown command "+sys.argv[i])
        

if __name__ == "__main__":
    main()
