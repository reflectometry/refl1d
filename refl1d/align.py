"""
Executable module for running refl1d profile alignment.

See :func:`refl1d.errors.run_errors` for details.
"""

# This needs to be in a module by itself that is not loaded by refl1d in
# order to allow the following on the command line::
#
#    $ refl1d -m refl1d.align ...
#
# We cannot put it in errors.py because that leads to a circular import
# when bumps.cli calls runpy.run_module. The module needs to be processed
# with run_module in order to execute the __name__ == "__main__" branch.

if __name__ == "__main__":
    from .errors import run_errors
    run_errors()
