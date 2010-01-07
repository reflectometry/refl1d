# This program is in the public domain
"""
Process monitors

Process monitors accept a history object each cycle and
perform some sort of work on it.
"""

class Monitor:
    """
    Generic monitor.
    """
    def config_history(self, history):
        """
        Indicate which fields are needed by the monitor and for what duration.
        """
        pass

    def __call__(self, history):
        """
        Give the monitor a new piece of history to work with.
        """
        pass

def _getfield(history, field):
    """
    Return the last value in the trace, or None if there is no
    last value or no trace.
    """
    trace = getattr(self, field, [])
    try:
        return trace[-1]
    except IndexError:
        return None

class Logger(Monitor):
    """
    Keeps a record of all values for the desired fields.

    Parameters::

        *fields*  ([strings] = [])
            The field names to use from the history.
        *table* (Table = None)
            An object with a store method that takes a series of key-value
            pairs, indexed by step=integer.

    Call logger.config_history(history) before starting so that the correct
    fields are stored.
    """
    def __init__(self, fields=[], table=None):
        self.fields = fields
        self.table = table
    def config_history(self, history):
        """
        Make sure history records the each logged field.
        """
        kwargs = dict((key,1) for key in self.fields)
        history.requires(**kwargs)
    def __call__(self, history):
        """
        Record the next piece of history.
        """
        record = dict((f,_getfield(history,f)) for f in self.fields)
        self.table.store(step=history.step,**record)


class TimedUpdates(Monitor):
    def __init__(self, progress_delta=50, improvement_delta=5):
        self.progress_delta = progress_delta
        self.improvement_delta = improvement_delta
        self.progress_time = None
        self.improvement_time = None

    def config_history(self, history):
        history.requires(time=1, value=1)

    def __call__(self, history):
        if len(history.time) == 0:
            history.time
