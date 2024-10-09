def clip(a, lo, hi):
    return max(lo, min(a, hi))


def setpar(p, v):
    p.value = clip(v, *p.prior.limits)


def CopyImage(canvas):
    """
    Copy the MPL canvas to the wx clipboard.
    """
    import wx

    bmp = wx.BitmapDataObject()
    bmp.SetBitmap(canvas.bitmap)

    wx.TheClipboard.Open()
    wx.TheClipboard.SetData(bmp)
    wx.TheClipboard.Close()


def showErrorMsg(parent, msg, title):
    """
    Show error message
    """
    import wx

    msg = wx.MessageDialog(parent, msg, title, wx.ICON_ERROR | wx.OK)
    msg.ShowModal()
    msg.Destroy()


def showWarningMsg(parent, msg, title):
    """
    Show Warning message
    """
    import wx

    msg = wx.MessageDialog(parent, msg, title, wx.ICON_WARNING | wx.OK)
    msg.ShowModal()
    msg.Destroy()


def CheckValid(s):
    """
    Translate the string into float, and check it.
    """
    try:
        val = float(s)
    except:
        val = 1.0
        msg = "%s can't change to float" % (s)
        raise ValueError(msg)

    return val


def twinx(ax=None):
    """
    Make a second axes overlay ax
    (or the current axes if ax is None) sharing the xaxis.

    The ticks for ax2 will be placed on the right,
    and the ax2 instance is returned.  See examples/two_scales.py

    Warning: This is a function to simulate the pylab.twinx in WX
    """
    if ax is None:
        return None

    ax2 = ax.figure.add_axes(ax.get_position(), sharex=ax, frameon=False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax.yaxis.tick_left()

    return ax2


def clear_axes(ax):
    """
    Clear axes from a figure
    """
    ax.legend_ = None
    ax.set_title(r"")
    ax.set_xlabel(r"")
    ax.set_ylabel(r"")
    ax.figure.canvas.draw_idle()


def filterP(f):
    """
    Translate a obj parameter into "string" format
    """
    if hasattr(f, "build"):
        return f.build()
    else:
        return f


def decodeP(p):
    """
    Translate a parameter into obj format

    Decompose a parameter.
    In this code. It only works for spline obj
    """
    try:
        ret = eval(p)
    except:
        ret = p
    return ret
