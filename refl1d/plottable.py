# DEAD CODE
class Vector(object):
    """
    Labelled set of numbers, possibly with uncertainty.

    This will usually be used for graphing.
    """
    def __init__(self, data=(), err=(), name="", units=""):
        self.name = name
        self.units = units
        self.data = data
        self.err = err

    def plot(self, page, **kw):
        return page.bar(self, **kw)

class Data(object):
    """
    A simple 1-D dataset x vs. y.
    """
    def __init__(self, x, y):
        self.x, self.y = x, y

    def plot(self, page, **kw):
        return page.marker(x, y, **kw)

class Trend(object):
    """
    A line fx vs. fy.  Note that this may be infinite extent, and
    should probably have a mechanism for recalculating given new
    axes limits.
    """
    def __init__(self, x=None, y=None):
        self.fx, self.fy = fx, fy

    def plot(self, page, **kw):
        return page.line(fx, fy, **kw)

class Histogram(object):
    def __init__(self, x=None):
        self.x = x

    def plot(self, page, **kw):
        return huh

class Fit(object):
    """
    Data plus trend.
    """
    def __init__(self, data=None, trend=None):
        self.data, self.trend = data, trend

    def plot(self, page):
        h1 = self.data.plot(page)
        h2 = self.trend.plot(page, style=h.style.complement())
        return huh

class Data2D(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def plot(self, page):
        page.surface(x, y, z)


# Colors need to complement or contrast with existing colors.
class Style(object):
    def complement(self, k=1, n=1):
        return Complement(self, k, n)

    def contrast(self):
        return Contrast(self)

class Complement(Style):
    def __init__(self, base, k, n):
        self.base, self.k, self.n = base, k, n

    def color(self):
        return self._color.darken(k, n)

class Contrast(Style):
    def __init__(self, base, k, n):
        self.base, self.k, self.n = base, k, n

    def color(self):
        return palette.next_color()

class Page(object):
    def marker(self, x=None, y=None, **kw):
        """
        Add markers for x, y to the axes.  If x, y have error bars, these
        will be included.  Legends are automatic.

        Returns a handle.
        """

    def line(self, x=None, y=None, **kw):
        """
        Add lines for x, y to the axes.  If y has error bars, these will
        be drawn appropriately.  Uncertainty in x will be ignored.

        Returns a handle.
        """

    def surface(self, x=None, y=None, z=None,  **kw):
        """
        Add a surface to the axes, with appropriate representation for
        uncertainty in x, y and z.

        Returns a handle.
        """

    def labels(self, x, y, text, **kw):
        """
        Add labels to the axes.
        """

class MplPage(Page):
    def _new_axes(self, x, y):
        pass

    def _find_axes(self, x, y):
        pass

    def _get_axes(self, x, y):
        ax = self._find_axes(x, y)
        if ax is None:
            ax = self._new_axes(x, y)
        return ax

    def marker(self, x=None, y=None, **kw):
        ax = self._get_axes(x, y)
        ax.plot(x.data, y.data)

    def line(self, x=None, y=None, **kw):
        ax = self._get_axes(x, y)
        ax.plot(x.data, y.data, '-')

    def surface(self, x=None, y=None, z=None, **kw):
        ax = self._get_axes(x, y)
        ax.pcolormesh(x.data, y.data, z.data)
