from copy import deepcopy

class SerialMapper(object):
    @staticmethod
    def start_worker(problem):
        pass
    @staticmethod
    def start_mapper(problem, modelargs):
        return lambda points: map(problem.nllf, points)
    @staticmethod
    def stop_mapper(mapper):
        pass

def _MP_set_problem(problem):
    global _problem
    _problem = problem
def _MP_run_problem(point):
    global _problem
    return _problem.nllf(point)
class MPMapper(object):
    @staticmethod
    def start_worker(problem):
        pass
    @staticmethod
    def start_mapper(problem, modelargs, cpus=None):
        import multiprocessing
        if cpus is None:
            cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus,_MP_set_problem,(problem,))
        mapper = lambda points: pool.map(_MP_run_problem, points)
        return mapper
    @staticmethod
    def stop_mapper(mapper):
        pass

class AMQPMapper(object):

    @staticmethod
    def start_worker(problem):
        #sys.stderr = open("dream-%d.log"%os.getpid(),"w")
        #print >>sys.stderr,"worker is starting"; sys.stdout.flush()
        from amqp_map.config import SERVICE_HOST
        from amqp_map.core import connect, start_worker as serve
        server = connect(SERVICE_HOST)
        #os.system("echo 'serving' > /tmp/map.%d"%(os.getpid()))
        #print "worker is serving"; sys.stdout.flush()
        serve(server, "dream", problem.nllf)
        #print >>sys.stderr,"worker ended"; sys.stdout.flush()

    @staticmethod
    def start_mapper(problem, modelargs):
        import multiprocessing
        from amqp_map.config import SERVICE_HOST
        from amqp_map.core import connect, Mapper

        server = connect(SERVICE_HOST)
        mapper = Mapper(server, "dream")
        cpus = multiprocessing.cpu_count()
        pipes = []
        for _ in range(cpus):
            cmd = [sys.argv[0], "--worker"] + modelargs
            #print "starting",sys.argv[0],"in",os.getcwd(),"with",cmd
            pipe = subprocess.Popen(cmd, universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pipes.append(pipe)
        for pipe in pipes:
            if pipe.poll() > 0:
                raise RuntimeError("subprocess returned %d\nout: %s\nerr: %s"
                                   % (pipe.returncode, pipe.stdout, pipe.stderr))
        #os.system(" ".join(cmd+["&"]))
        import atexit
        def exit_fun():
            for p in pipes: p.terminate()
        atexit.register(exit_fun)

        #print "returning mapper",mapper
        return mapper

    @staticmethod
    def stop_mapper(mapper):
        for pipe in mapper.pipes:
            pipe.terminate()
