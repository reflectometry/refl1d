
Using the posterior distribution
================================

You can reload an uncertainty analysis after the fact::

    $ ipython -pylab

    >>> import dream.state
    >>> state = dream.state.load_state(modelname)
    >>> state.mark_outliers() # ignore outlier chains
    >>> state.show()  # Plot statistics


You can restrict a variable to a certain range when doing plots.
For example, to restrict the third parameter to [0.8-1.0] and the
fifth to [0.2-0.4]::

    >>> from dream import views
    >>> selection={2: (0.8,1.0), 4:(0.2,0.4),...}
    >>> views.plot_vars(state, selection=selection)
    >>> views.plot_corrmatrix(state, selection=selection)

You can also add derived variables using a function to generate the
derived variable.  For example, to add a parameter which is p[0]+p[1]
use::

    >>> state.derive_vars(lambda p: p[0]+p[1], labels=["x+y"])

You can generate multiple derived parameters at a time with a function
that returns a sequence::


    >>> state.derive_vars(lambda p: (p[0]*p[1],p[0]-p[1]), labels=["x*y","x-y"])

These new parameters will show up in your plots::

    >>> state.show()

The plotting code is somewhat complicated, and matplotlib doesn't have a
good way of changing plots interactively.  If you are running directly
from the source tree, you can modify the dream plotting libraries as you
need for a one-off plot, the replot the graph::


    # ... change the plotting code in dream.views/dream.corrplot
    >>> reload(dream.views)
    >>> reload(dream.corrplot)
    >>> state.show()

Be sure to restore the original versions when you are done.  If the change
is so good that everyone should use it, be sure to feed it back to the
community via http://github.com/reflecometry/refl1d.
