
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
    map_queue = ".".join(("map-J",jobid))
    channel.exchange_declare(exchange=exchange, type="direct",
                             durable=False, auto_delete=True)
    channel.queue_declare(queue=map_queue, durable=False, 
                          exclusive=False, auto_delete=True)

    # Prefetch requires basic_ack, basic_qos and consume with ack
    def _process_work(msg):
        # Check for sentinel
        if msg.reply_to == "": channel.basic_cancel(consumer)
        body = loads(msg.body)
        # Acknowledge delivery of message
        #print "processing...",body['index'],body['value']; sys.stdout.flush()
        try:
            result = work(body['value'])
        except Exception,exc:
            result = None
        #print "done"
        channel.basic_ack(msg.delivery_tag)
        reply = amqp.Message(dumps(dict(index=body['index'],result=result)))
        channel.basic_publish(reply, exchange=exchange, 
                              routing_key=msg.reply_to)
    #channel.basic_qos(prefetch_size=0, prefetch_count=1, a_global=False)
    consumer = channel.basic_consume(queue=map_queue, callback=_process_work, 
                                     no_ack=False)
    #print "kernel waiting on",map_queue,"with",work
    while True:
        channel.wait()

class Mapper(object):
    def __init__(self, server, jobid):
        # Create the exchange and the worker and reply queues
        channel = server.channel()
        exchange = "park.map"
        channel.exchange_declare(exchange=exchange, type="direct",
                                 durable=False, auto_delete=True)

        map_channel = channel
        map_queue = ".".join(("map-J",jobid))
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
        #TODO: proper shutdown of consumers
        for i in range(1000):
            msg = amqp.Message("", reply_to="",delivery_mode=1)
            self.map_channel.basic_publish(msg, exchange=self.exchange,
                                           routing_key=self.map_queue)
        self.map_channel.close()
        self.reply_channel.close()
    def _process_result(self, msg):
        self._reply = loads(msg.body)
        #print "received result",self._reply['index'],self._reply['result']
    @daemon
    def _send_map(self, items):
        for i,v in enumerate(items):
            self.num_queued = i
            #print "queuing %d %s"%(i,v)
            
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
        Stop a running map.
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
            #print "received %d %g"%(idx,result)
            self.num_processed += 1

            ## USE_LOCKS_TO_THROTTLE
            if self.num_queued - self.num_processed < config.MAX_QUEUE - 10:
                # Ten at a time go through for slow processes
                self._throttle.acquire()
                self._throttle.notify()
                self._throttle.release() 
            
            yield idx,result
        publisher.join()
    def __call__(self, items):
        result = list(self.async(items))
        result = list(sorted(result,lambda x,y: cmp(x[0],y[0])))
        return zip(*result)[1]

