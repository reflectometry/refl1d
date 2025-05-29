#!/usr/bin/env python
"""
Run bumps with refl1d plugin.

The *align* subcommand can be used on a completed DREAM fit to redraw the
profile contours aligned to a different layer boundary.
See :func:`refl1d.uncertainty.run_errors` for details.

** DEPRECATED **
"""

import sys

from . import __version__


def cli():
    """
    Install the Refl1D plugin into bumps and run the command line interface.
    """
    import bumps.plugin

    bumps.plugin.ACTIVE_PLUGIN_NAME = "refl1d"
    # TODO: Add subcommand support to bumps.
    if len(sys.argv) > 2 and sys.argv[1] == "align":
        from .uncertainty import run_errors

        del sys.argv[1]
        run_errors()
    else:
        import bumps.cli

        bumps.cli.main()


def gui():
    """
    Install the Refl1D plugin into bumps and run the graphical user interface.
    """
    import bumps.plugin
    import bumps.gui.gui_app

    bumps.plugin.ACTIVE_PLUGIN_NAME = "refl1d"
    bumps.gui.gui_app.main()


if __name__ == "__main__":
    cli()
