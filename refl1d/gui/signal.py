import wx
from wx.py.dispatcher import send, connect

def model_new(model):
    send('model.new',model=model)

def update_model(model, dirty=True):
    # signaller is responsible for marking the model as needing recalculation
    if dirty: model.model_update()
    send('model.update_structure',model=model)

_DELAYED_SIGNAL = {}
def update_parameters(model, delay=50):
    """
    Inform all views that the model has changed.  Note that if the model
    is changing rapidly, then the signal will be delayed for a time.
    """
    # signaller is responsible for marking the model as needing recalculation
    model.model_update()
    try:
        _DELAYED_SIGNAL[model].Restart(delay)
    except:
        def _send_signal():
            #print "sending update parameters",model
            del _DELAYED_SIGNAL[model]
            send('model.update_parameters',model=model)
        _DELAYED_SIGNAL[model] = wx.FutureCall(delay, _send_signal)

def log_message(message):
    send('log',message=message)
