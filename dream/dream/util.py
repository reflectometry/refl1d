def console():
    """
    Start the python console with the local variables available.
    
    console() should be the last thing in the file, after sampling and
    showing the default plots.
    """
    import os
    import sys

    # Hack for eclipse console: can't actually run ipython in the eclipse
    # console and get it to plot, so instead guess whether we are in a 
    # console by checking if we are attached to a proper tty through stdin.
    # For eclipse, just show the plots.
    try:
        tty = os.isatty(sys.stdin.fileno())
    except:
        tty = False

    if tty:
        # Display outstanding plots and turn interactive on
        from matplotlib import interactive
        from matplotlib._pylab_helpers import Gcf
        for fig in Gcf.get_all_fig_managers():
            try: # CRUFT
                fig.show()
            except:
                fig.frame.Show()
        interactive(True)

        # Start an ipython shell with the caller's local variables
        import IPython
        symbols = sys._getframe(1).f_locals
        ip = IPython.Shell.IPShell(user_ns=symbols)
        ip.mainloop()
    else:
        # Not a tty; try doing show() anyway
        import pylab
        pylab.show()
