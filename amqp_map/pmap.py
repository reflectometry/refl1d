raise NotImplementedError  # This code is still a work in progress
from .rpc import RPCMixin

def start_generic(server):
    """
    Client side driver of the map work.

    The model should already be loaded before calling this.
    """
    # Create the exchange and the worker queue
    channel = server.channel()
    exchange = "park.map"
    map_queue = ".".join(("map",mapid))
    channel.exchange_declare(exchange=exchange, type="direct",
                             durable=False, auto_delete=True)
    channel.queue_declare(queue=map_queue, durable=False, 
                          exclusive=False, auto_delete=True)



    cache = {}
    def _lookup(function):
        if function not in cache:
            
            
        
    # Prefetch requires basic_ack, basic_qos and consume with ack
    def _process_work(msg):
        # Check for sentinel
        if msg.reply_to == "": channel.basic_cancel(consumer)
        
        fn = _lookup(msg.function)        
        
        body = loads(msg.body)
        # Acknowledge delivery of message
        #print "processing...",body['index'],body['value'] 
        try:
            result = function(body['value'])
        except exc:
            result = None
        #print "done"
        channel.basic_ack(msg.delivery_tag)
        reply = amqp.Message(dumps(dict(index=body['index'],result=result)))
        channel.basic_publish(reply, exchange=exchange, 
                              routing_key=msg.reply_to)
    #channel.basic_qos(prefetch_size=0, prefetch_count=1, a_global=False)
    consumer = channel.basic_consume(queue=map_queue, callback=_process_work, 
                                     no_ack=False)
    while True:
        channel.wait()



class Mapper(object, RPCMixin):
    def server(self, server):
        # Create the exchange and the worker and reply queues
        channel = server.channel()
        exchange = "park.map"
        channel.exchange_declare(exchange=exchange, type="direct",
                                 durable=False, auto_delete=True)

        map_channel = channel
        map_queue = ".".join(("map",mapid))
        map_channel.queue_declare(queue=map_queue, durable=False, 
                                  exclusive=False, auto_delete=True)
        map_channel.queue_bind(queue=map_queue, exchange="park.map",
                               routing_key = map_queue)
        
        reply_channel = server.channel()
        #reply_queue = ".".join(("reply",mapid)) # Fixed Queue name
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
        self.channel.close()

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

    def _cache(self, fn):
        self._function_str = dumps(fn)
        self._function_id = md5sum(self._function_str)

    def _process_rpc(self, msg):
        self._reply = loads(msg.body)
        #print "received result",self._reply['index'],self._reply['result']
    @daemon
    def _rpc_handler(self, items):
        while True:
            self.rpc_channel.wait()

    # RPC call
    def get_function(self, id):
        if id == self._function_id:
            return self._function_str
        else:
            raise ValueError("Function not being mapped")

    def cancel(self):
        """
        Stop a running map.
        """
        raise NotImplementedError()
        # Need to clear the queued items and notify async that no more results.
        # Messages in transit need to be ignored, which probably means tagging
        # each map header with a call number so that previous calls don't
        # get confused with current calls.
        self.reply_channel.basic_publish(msg)

    def async(self, fn, items):
        self._cache(fn)
        items = list(items) # make it indexable
        self.num_items = len(items)
        # Queue items in separate thread so we can start receiving results
        # before all items are even queued
        self.num_processed = 0
        publisher = self._send_map(items)
        for i in items:
            self.reply_channel.wait()
            idx = self._reply['index']
            result = self._reply['result']
            #print "received %d %g"%(idx,result)
            self.num_processed = i

            ## USE_LOCKS_TO_THROTTLE
            if self.num_queued - self.num_processed < config.MAX_QUEUE - 10:
                # Ten at a time go through for slow processes
                self._throttle.acquire()
                self._throttle.notify()
                self._throttle.release()
            
            yield idx,result
        publisher.join()
    def __call__(self, fn, items):
        result = list(self.async(fn, items))
        result = list(sorted(result,lambda x,y: cmp(x[0],y[0])))
        return zip(*result)[1]

map = PickleMapper()


