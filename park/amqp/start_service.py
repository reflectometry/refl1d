from park.environment import Environment
from park.store import Store
from park.amqp.config import SERVICE_HOST
from park.amqp.core import connect, Mapper

def start(jobid, module, init):
    server = connect(SERVICE_HOST)
    mapper = Mapper(server, jobid)
    env = Environment(mapper=mapper, store=Store())
    env.run_service(jobid, module, init)

if __name__ == "__main__":
    import sys
    start(*sys.argv[1:])
