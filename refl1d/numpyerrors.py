"""
With context for handling numpy errors.

Use this when you have a routine with numeric issues such as divide by zero
which are known to be harmless, for example, because infinite or NaN results
are allowed by the interface, or because the remainder of the code accommodates
the exceptional conditions.

Usage
-----

This is a wrapper around the numpy.seterr() command, and uses the same
types of error handling controls, but in a with context or as a decorator::

    with Errors(...):
        statements

    @errors(...)
    def f():
        statements

The arguments to Errors and errors are identical to numpy.seterr.

Some convenience decorators are predefined: ignored, raised, printed, warned.

Example
-------

    >>> import numpy
    >>> with Errors(all='ignore'): x = 1/numpy.zeros(3)
    >>> with Errors(all='print'): x = 1/numpy.zeros(3)  # doctest:+SKIP
    Warning: divide by zero encountered in divide
    >>> @ignored
    ... def f(): x = 1/numpy.zeros(3)
    >>> f()
    >>> @printed
    ... def g(): x = 1/numpy.zeros(3)
    >>> g() # doctest:+SKIP
    Warning: divide by zero encountered in divide
"""
from __future__ import with_statement
import functools
import numpy
class Errors(object):
    def __init__(self, **kw):
        self.desired_state = kw
    def __enter__(self):
        self.current_state = numpy.geterr()
        numpy.seterr(**self.desired_state)
    def __exit__(self, *args):
        numpy.seterr(**self.current_state)
def errors(**kw):
    context = Errors(**kw)
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kw):
            with context:
                return f(*args, **kw)
        return decorated
    return decorator
ignored = errors(all='ignore')
raised = errors(all='raise')
printed = errors(all='print')
warned = errors(all='warn')
