import wx
import string

# This is a workaround for a py2exe problem when using pubsub from wxPython ...
# wxPython 2.8.11.0 supports both the pubsub V1 and V3 APIs (V1 is the default;
# there is no V2), whereas 2.8.9.x and 2.8.10.1 offer only the V1 API. Since
# we want to build with older versions of wxPython we will use the V3 API,
# but provide a shim so that we can transport options using the V1 interface.
try:
    # Check if V3 interface is available
    from wx.lib.pubsub import setupkwargs
    from wx.lib.pubsub import pub
    subscribe = pub.subscribe
    publish = pub.sendMessage
    #print "using V3 interface to pubsub"
except:
    #print "using V1 interface to pubsub"
    # Otherwise use the V1 interface
    from wx.lib.pubsub import Publisher
    _pub = Publisher()
    _subscribers = [] # pubsub uses weak refs; need to hold on to subscribers
    def subscribe(callback, topic):
        def unwrap_data(event):
            #print "recving",topic,"in",callback.__name__
            callback(**event.data)
        _subscribers.append(unwrap_data)
        #print "subscribe",callback.__name__,"to",topic
        _pub.subscribe(unwrap_data, topic)
    def publish(topic, **kwargs):
        #print "sending",topic
        _pub.sendMessage(topic, kwargs)

class Validator(wx.PyValidator):
    def __init__(self, flag):
        wx.PyValidator.__init__(self)
        self.flag = flag
        self.Bind(wx.EVT_CHAR, self.OnChar)
    def Clone(self):
        return Validator(self.flag)
    def Validate(self, win):
        return True
    def TransferToWindow(self):
        return True
    def TransferFromWindow(self):
        return True
    def OnChar(self, evt):
        key = chr(evt.GetKeyCode())
        if self.flag == "no-alpha" and key in string.letters:
            return
        if self.flag == "no-digit" and key in string.digits:
            return
        evt.Skip()

def nice(v, digits = 4):
    from math import log, log10, ceil, floor
    """Fix v to a value with a given number of digits of precision"""
    if v == 0.: return v
    sign = v/abs(v)
    place = floor(log10(abs(v)))
    scale = 10**(place-(digits-1))
    return sign*floor(abs(v)/scale+0.5)*scale
