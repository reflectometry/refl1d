#!/usr/bin/env python
"""
Run bumps with refl1d plugin
"""
from . import __version__

def setup_bumps():
    """
    Install the refl1d plugin into bumps, but don't run main.
    """
    import bumps.cli
    bumps.cli.set_mplconfig(appdatadir='Refl1D-'+__version__)
    from . import fitplugin
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
