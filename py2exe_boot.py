# This program is public domain

# Redirect Stderr to a message box rather than trying to write to the
# unwritable app directory.  Add this to setup_exe as:
#
#    setup(
#        options = {
#             "py2exe": {
#                 ...,
#                 "custom_boot_script": "py2exe_boot.py",
#              }
#         }
#    )
#
# Note: this doesn't seem to be working, but leave it in for now.
# Haven't confirmed that it is actually hooked in.
#
# Based on suggestion by: Marko Loparic
# py2exe-users@lists.sourceforge.net [1 Sep 17:13 2011]

import sys

if sys.frozen == "windows_exe":

    class Stderr(object):
        softspace = 0
        _file = None
        _alert = sys._MessageBox  # used atexit, so keep a handle here

        def _display_error(self):
            text = "Captured stderr:\n" + self._file.getvalue()
            self._alert(0, text)

        def write(self, text):
            if self._file is None:
                from StringIO import StringIO

                self._file = StringIO()
                import atexit

                atexit.register(self._display_error)
            self._file.write(text)

        def flush(self):
            pass

    sys.stderr = Stderr()
    del Stderr
del sys
