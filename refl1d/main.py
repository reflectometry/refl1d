#!/usr/bin/env python
"""
Run bumps with refl1d plugin
"""


from . import fitplugin

def install_plugin():
    """
    Install the refl1d plugin into bumps, but don't run main.
    """
    import bumps.cli
    bumps.cli.install_plugin(fitplugin)

def cli():
    """
    Install the Refl1D plugin into bumps and run the command line interface.
    """
    import bumps.cli
    install_plugin()
    bumps.cli.main()

def gui():
    """
    Install the Refl1D plugin into bumps and run the graphical user interface.
    """
    import bumps.gui.gui_app
    install_plugin()
    bumps.gui.gui_app.main()

if __name__ == "__main__":
    cli()
