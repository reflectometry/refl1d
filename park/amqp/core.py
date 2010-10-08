
# Mechanisms for throttling the mapper when the function is expensive:
#
# 1) Do nothing.
#    PRO: no computation overhead
#    PRO: AMQP will use flow control when it runs low on memory
#    PRO: most maps are small
#    CON: may use excess memory on exchange
# 2) Use locks.
#    PRO: threading.Condition() makes this easy
#    PRO: good lock implementation means master consumes no resources
#    CON: may not handle keyboard interrupt correctly on some platforms
# 3) Use sleep.
#    PRO: simple implementation will work across platforms
#    CON: master stays in memory because it is restarted every 10 min.
#
# The current implementation uses locks to throttle.

## USE_LOCKS_TO_THROTTLE
import threading

## USE_SLEEP_TO_THROTTLE
#import time

#from dill import loads, dumps
from cPickle import loads, dumps
import sys

from amqplib import client_0_8 as amqp

from . import config
from .url import URL
import time
from .threaded import threaded, daemon

def connect(url, insist=False):
    url = URL(url, host="localhost", port=5672,
              user="guest", password="guest", path="/")
    host = ":".join( (url.host, str(url.port)) )
    userid,password = url.user,url.password
    virtual_host = "/" + url.path
    server = amqp.Connection(host=host, userid=userid, password=password,
                             virtual_host=virtual_host, insist=insist)
    return server

def start_kernel(server, jobid, work):
    """
    Client side driver of the map work.

    The model should already be loaded before calling this.
    """
    # Create the exchange and the worker queue
    channel = server.channel()
    exchange = "park.map"
    map_queue = "".join(("park.map.J",jobid))
    channel.exchange_declare(exchange=exchange, type="direct",
                             durable=False, auto_delete=True)
    channel.queue_declare(queue=map_queue, durable=False,
                          exclusive=False, auto_delete=True)

    # Prefetch requires basic_ack, basic_qos and consume with ack
    def _process_work(msg):
        # Check for sentinel
        if msg.reply_to == "":
            #print "Done mapping"
            channel.basic_cancel(map_queue)
            # TODO: this is too brutal
            sys.exit()
            return
        body = loads(msg.body)
        # Acknowledge delivery of message
        #print "processing...",body['index'],body['value']; sys.stdout.flush()
        try:
            result = work(body['value'])
        except Exception,exc:
            # TODO: do we really want to ignore kernel errors?
            result = None
        #print "done"
        channel.basic_ack(msg.delivery_tag)
        reply = amqp.Message(dumps(dict(index=body['index'],result=result)))
        channel.basic_publish(reply, exchange=exchange,
                              routing_key=msg.reply_to)
    #channel.basic_qos(prefetch_size=0, prefetch_count=1, a_global=False)
    channel.basic_consume(queue=map_queue, callback=_process_work,
                          no_ack=False, consumer_tag=map_queue)
    #print "kernel waiting on",map_queue,"with",work
    while True:
        channel.wait()

class Mapper(object):
    """
    Map work items to workers using a message server and a job-specific queue.

    *server* is a handle to an open amqp server.

    *jobid* is the id to use for the messages.
    """
    def __init__(self, server, jobid):
        # Create the exchange and the worker and reply queues
        channel = server.channel()
        exchange = "park.map"
        channel.exchange_declare(exchange=exchange, type="direct",
                                 durable=False, auto_delete=True)

        map_channel = channel
        map_queue = "".join(("park.map.J",jobid))
        map_channel.queue_declare(queue=map_queue, durable=False,
                                  exclusive=False, auto_delete=True)
        map_channel.queue_bind(queue=map_queue, exchange="park.map",
                               routing_key = map_queue)

        reply_channel = server.channel()
        #reply_queue = ".".join(("reply",jobid)) # Fixed Queue name
        reply_queue = "" # Let amqp create a temporary queue for us
        reply_queue,_,_ = reply_channel.queue_declare(queue=reply_queue,
                                                      durable=False,
                                                      exclusive=True,
                                                      auto_delete=True)
        reply_channel.queue_bind(queue=reply_queue, exchange="park.map",
                                 routing_key = reply_queue)
        reply_channel.basic_consume(queue=reply_queue,
                                    callback=self._process_result,
                                    no_ack=True)
        self.exchange = exchange
        self.map_queue = map_queue
        self.map_channel = map_channel
        self.reply_queue = reply_queue
        self.reply_channel = reply_channel

        ## USE_LOCKS_TO_THROTTLE
        self._throttle = threading.Condition()

    def close(self):
        """
        Terminate the map, and close the queues.
        """
        #TODO: proper shutdown of consumers
        #print "closing"; sys.stdout.flush()
        for i in range(1000):
            msg = amqp.Message("", reply_to="",delivery_mode=1)
            self.map_channel.basic_publish(msg, exchange=self.exchange,
                                           routing_key=self.map_queue)
        self.map_channel.close()
        self.reply_channel.close()
    def _process_result(self, msg):
        """
        Run in background, receiving results from the workers.
        """
        self._reply = loads(msg.body)
        #print "received result",self._reply['index'],self._reply['result']
    @daemon
    def _send_map(self, items):
        """
        Run in background, posting new items to be mapped as results
        come available.
        """
        # TODO: need map call number in addition to item number in order
        # to handle cancel properly.
        for i,v in enumerate(items):
            self.num_queued = i
            #print "queuing %d %s"%(i,v); sys.stdout.flush()

            ## USE_LOCKS_TO_THROTTLE
            if  self.num_queued - self.num_processed > config.MAX_QUEUE:
                #print "sleeping at %d in %d out"%(i,self.num_processed)
                self._throttle.acquire()
                self._throttle.wait()
                self._throttle.release()
                #print "waking at %d in %d out"%(i,self.num_processed)

            # USE_SLEEP_TO_THROTTLE
            #sleep_time = 0.2
            #while i - self.num_processed > config.MAX_QUEUE:
            #    #print "sleeping %g with in=%d out=%d"%(sleep_time,self.num_queued,self.num_processed)
            #    time.sleep(sleep_time)
            #    sleep_time = min(2*sleep_time, 600)

            body = dumps(dict(index=i,value=v))
            msg = amqp.Message(body, reply_to=self.reply_queue, delivery_mode=1)
            self.map_channel.basic_publish(msg, exchange=self.exchange,
                                           routing_key=self.map_queue)

    def cancel(self):
        """
        Stop a running map.  ** Not Implemented **
        """
        raise NotImplementedError()
        # Need to clear the queued items and notify async that no more results.
        # Messages in transit need to be ignored, which probably means tagging
        # each map header with a call number so that previous calls don't
        # get confused with current calls.
        msg = amqp.Message("", reply_to="", delivery_mode=1)
        self.map_channel.basic_publish(msg, exchange=self.exchange,
                                       routing_key=self.map_queue)

    def async(self, items):
        """
        Return i,f(v) for i,v in enumerate(items) in the order that they
        are received.  The received order is not necessarily the order in
        which the items occur in the list.
        """
        #print "starting map"; sys.stdout.flush()
        # TODO: we should be able to flag completion somehow so that the
        # whole list does not need to be formed.
        items = list(items) # make it indexable
        self.num_items = len(items)
        # Queue items in separate thread so we can start receiving results
        # before all items are even queued
        self.num_processed = 0
        publisher = self._send_map(items)
        recvd = set()
        while self.num_processed < self.num_items:
            self.reply_channel.wait()
            idx = self._reply['index']
            if idx in recvd: continue
            recvd.add(idx)
            result = self._reply['result']
            #print "received %d %g"%(idx,result); sys.stdout.flush()
            self.num_processed += 1

            ## USE_LOCKS_TO_THROTTLE
            if self.num_queued - self.num_processed < config.MAX_QUEUE - 10:
                # Ten at a time go through for slow processes
                self._throttle.acquire()
                self._throttle.notify()
                self._throttle.release()

            yield idx,result
        publisher.join()

    def imap(self, items):
        """
        Return f(v) for v in items in order as they become available.
        """
        complete = {}
        next_index = 0
        for i,v in self.async(items):
            if i == next_index:
                yield v
                next_index += 1
            else:
                complete[i] = v
            while next_index in complete:
                yield complete[next_index]
                del complete[next_index]
                next_index += 1

    def map(self, items):
        """
        Return [f(v) for v in items]
        """
        result = list(self.async(items))
        result = list(sorted(result,lambda x,y: cmp(x[0],y[0])))
        return zip(*result)[1]

    __call__ = map
