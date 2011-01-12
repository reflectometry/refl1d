"""
Yet another dispatcher.

Listeners register with (trigger,value).
Generators signal with (trigger,value).
"""

class Listener(object):
    def __init__(self):
        self.connection = {}

    def connect(self,trigger,value,callback):
        key = (trigger,value)
        if key in self.connection:
            self.connection[key].append(callback)
        else:
            self.connection[key] = [callback]

    def disconnect(self,trigger,value,callback):
        key = (trigger,value)
        if key in self.connection and callback in self.connection[key]:
            self.connection[key].remove(callback)

    def signal(self,trigger,value,**kw):
        key = (trigger,value)
        for callback in self.connection.get(key,[]):
            callback(trigger,value,**kw)
