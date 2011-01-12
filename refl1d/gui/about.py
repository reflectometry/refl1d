# Copyright (C) 2006-2010, University of Maryland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Author: James Krycka

"""
This module contains a custom About Dialog class and associated text strings
used for informational display purposes.  Note that the product version is
maintained in the version.py file and thus imported here.
"""

import os
import wx

try:
    from agw.hyperlink import HyperLinkCtrl
except ImportError: # if it's not there, try the older location.
    from wx.lib.agw.hyperlink import HyperLinkCtrl

from wx.lib.wordwrap import wordwrap

from version import version as APP_VERSION
from common.utilities import get_appdir

# Resource files.
PROG_ICON = "ksrefl.ico"

# Text strings used in About Dialog boxes and for other project identification
# purposes.
APP_NAME = "Refl1D"

APP_TITLE = "Reflectometry-1D"

APP_COPYRIGHT = "(C) 2010 University of Maryland"

APP_DESCRIPTION = """\
The KsRefl Depth Profile Modeller provides an interactive user interface for \
modelling an experimental sample and generating a scattering length density \
(SLD) profile of the layers in the sample.  Using 1D models, reflectometry \
data sets from either polarized or unpolarized neutron scattering experiments, \
and a fitting service, KsRefl calculates and plots the depth profile of the \
material which can be composed of a mixture of thin film and continuously \
varying free form layers.
"""

APP_LICENSE = """\
Permission is hereby granted, free of charge, to any person obtaining a copy \
of this software and associated documentation files (the "Software"), to deal \
in the Software without restriction, including without limitation the rights \
to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell \
copies of the Software, and to permit persons to whom the Software is \
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in \
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR \
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, \
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE \
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER \
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, \
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN \
THE SOFTWARE.
"""

APP_CREDITS = """\
This program was developed jointly by the University of Maryland (UMD) and \
the National Institute of Standards and Technology (NIST) as part of the \
Distributed Data Analysis of Neutron Scattering Experiments (DANSE) project \
funded by the US National Science Foundation under grant DMR-0520547.

Paul Kienzle, NIST
    - Application concept
    - Application API
    - Reflectivity and resolution calculation
James Krycka, UMD
    - GUI development
Nikunj Patel, UMD
    - GUI development
"""

APP_PROJECT_URL = "http://reflectometry.org/danse"
APP_PROJECT_TAG = "DANSE/Reflectometry home page"

APP_TUTORIAL_URL = "http://www.reflectometry.org/danse/packages.html"
APP_TUTORIAL_TAG = "DANSE/Reflectometry documentation"
APP_TUTORIAL = """\
Note: the tutorial is not currently available.

For a tutorial and other documentation on KsRefl, please visit:\
"""

#==============================================================================

class AboutDialog(wx.Dialog):
    """
    This class creates a pop-up About Dialog box with several display options.
    """

    def __init__(self,
                 parent=None,
                 id = wx.ID_ANY,
                 title="About",
                 pos=wx.DefaultPosition,
                 size=wx.DefaultSize,
                 style=wx.DEFAULT_DIALOG_STYLE,
                 show_name=True,
                 show_notice=True,
                 show_link=True,
                 show_link_docs=False,
                 info="..."
                ):
        wx.Dialog.__init__(self, parent, id, title, pos, size, style)

        # Display the application's icon in the title bar.
        icon = wx.Icon(os.path.join(get_appdir(), PROG_ICON),
                       wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

        # Set the font for this window and all child windows (widgets) from the
        # parent window, or from the system defaults if no parent is given.
        # A dialog box does not inherit font info from its parent, so we will
        # explicitly get it from the parent and apply it to the dialog box.
        if parent is not None:
            font = parent.GetFont()
            self.SetFont(font)

        # Display program name and version.
        if show_name:
            prog = wx.StaticText(self, wx.ID_ANY,
                                 label=(APP_NAME + " " + APP_VERSION))
            font = prog.GetFont()
            font.SetPointSize(font.GetPointSize() + 1)
            font.SetWeight(wx.BOLD)
            prog.SetFont(font)

        # Display copyright notice.
        if show_notice:
            copyright = wx.StaticText(self, wx.ID_ANY, label=APP_COPYRIGHT)

        # Display hyperlink to the Reflectometry home page and/or doc page.
        if show_link:
            hyper1 = HyperLinkCtrl(self, wx.ID_ANY, label=APP_PROJECT_TAG,
                                                    URL=APP_PROJECT_URL)
        if show_link_docs:
            hyper2 = HyperLinkCtrl(self, wx.ID_ANY, label=APP_TUTORIAL_TAG,
                                                    URL=APP_TUTORIAL_URL)

        # Display the body of text for this about dialog box.
        info = wx.StaticText(self, wx.ID_ANY,
                             label=wordwrap(info, 530, wx.ClientDC(self)))
        # Create the OK button control.
        ok_button = wx.Button(self, wx.ID_OK, "OK")
        ok_button.SetDefault()

        # Use a vertical box sizer to manage the widget layout..
        sizer = wx.BoxSizer(wx.VERTICAL)
        if show_name:
            sizer.Add(prog, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, border=10)
        if show_notice:
            sizer.Add(copyright, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, border=10)
        sizer.Add(info, 0, wx.ALL, border=10)
        if show_link:
            sizer.Add(hyper1, 0, wx.ALL, border=10)
        if show_link_docs:
            sizer.Add(hyper2, 0, wx.ALL, border=10)
        sizer.Add(ok_button, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, border=10)

        # Finalize the sizer and establish the dimensions of the dialog box.
        self.SetSizer(sizer)
        sizer.Fit(self)
