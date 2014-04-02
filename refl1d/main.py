#!/usr/bin/env python
"""
Run bumps with refl1d plugin
"""

def set_mplconfig():
    """
    Point the matplotlib config dir to somewhere writeable
    """
    import os,sys
    if hasattr(sys, 'frozen'):
        mplconfigdir = os.path.join(os.environ['APPLOCALDATA'],'Refl1D', 'mplconfig')
        mplconfigdir = os.environ.setdefault('MPLCONFIGDIR',mplconfigdir)
        if not os.path.exists(mplconfigdir): os.makedirs(mplconfigdir)

def setup_bumps():
    """
    Install the refl1d plugin into bumps, but don't run main.
    """
    set_mplconfig()
    from . import fitplugin
    import bumps.cli
    bumps.cli.install_plugin(fitplugin)

def cli():
    """
    Install the Refl1D plugin into bumps and run the command line interface.
    """
    setup_bumps()
    import bumps.cli
    bumps.cli.main()

def gui():
    """
    Install the Refl1D plugin into bumps and run the graphical user interface.
    """
    setup_bumps()
    import bumps.gui.gui_app
    bumps.gui.gui_app.main()

if __name__ == "__main__":
    cli()
