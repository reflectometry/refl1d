"""
interface  interactor.
"""

from .config import interface_color, pick_radius
from .interactor import BaseInteractor
from .util import clip, setpar

MAX_ROUGH = 1


class InterfaceInteractor(BaseInteractor):
    """
    Control the roughness of the layers.
    """

    def __init__(self, profile):
        BaseInteractor.__init__(self, profile)
        ax = profile.axes

        # markers for roughness
        style = dict(
            linestyle="",
            transform=profile.xcoords,
            marker="s",  # square
            markersize=7,
            color=interface_color,
            alpha=0.6,
            pickradius=pick_radius,
            zorder=8,  # Prefer this to other lines
            visible=False,
        )
        self.markers = [
            ax.plot([0], [0.05], label=label, **style)[0] for label in ("interface marker L", "interface marker R")
        ]

        # lines for roughness
        style = dict(linestyle="-", transform=profile.xcoords, marker="", color=interface_color, visible=False)
        self.lines = [
            ax.plot([0, 0], [0.05, 0.05], label=label, **style)[0] for label in ("interface line L", "interface line R")
        ]

        self.connect_markers(self.markers)
        self._left = self._right = None

    def set_layer(self):
        """
        Move markers to the new layer
        """
        n = self.profile.layer_num
        z = self.profile.boundary[1:-1]
        show_left = n is not None and n > 0
        show_right = n is not None and n < len(z)
        if show_left:
            self._left = self.profile.sample_layer(n - 1).interface
        else:
            self._left = None
        if show_right:
            self._right = self.profile.sample_layer(n).interface
        else:
            self._right = None
        # self.update_markers()

    def update_markers(self):
        """
        Draw the new roughness on the graph.
        """
        n = self.profile.layer_num
        z = self.profile.boundary[1:-1]
        show_left = self._left is not None
        show_right = self._right is not None

        self.markers[0].set(visible=show_left)
        self.lines[0].set(visible=show_left)
        if show_left:
            self.markers[0].set_xdata([z[n - 1] + self._left.value])
            self.lines[0].set_xdata([z[n - 1], z[n - 1] + self._left.value])

        self.markers[1].set(visible=show_right)
        self.lines[1].set(visible=show_right)
        if show_right:
            self.markers[1].set_xdata([z[n] - self._right.value])
            self.lines[1].set_xdata([z[n], z[n] - self._right.value])

    def clear_markers(self):
        """
        Remove interface markers from the graph.
        """
        BaseInteractor.clear_markers(self)
        for h in self.lines:
            h.remove()
        self.lines = []

    def save(self, event):
        """
        Remember the interface for this layer and the next so that we
        can restore on Esc.
        """

        if self._left is not None:
            self._left_value = self._left.value
        if self._right is not None:
            self._right_value = self._right.value

    def restore(self, event):
        """
        Restore the roughness for this layer.
        """
        if self._left is not None:
            self._left.value = self._left_value
        if self._right is not None:
            self._right.value = self._right_value

    def drag(self, event):
        """
        Process move to a new position, making sure that the move is allowed.
        """
        z = self.profile.boundary
        n = self.profile.layer_num
        side = self.markers.index(event.artist)
        if side == 0:  # Left
            limit = min(z[n] - z[n - 1], z[n + 1] - z[n])
            v = clip(event.xdata - z[n], 0, limit / MAX_ROUGH)
            setpar(self._left, v)

        else:  # Right
            limit = min(z[n + 1] - z[n], z[n + 2] - z[n + 1])
            v = clip(z[n + 1] - event.xdata, 0, limit / MAX_ROUGH)
            setpar(self._right, v)

        # self.update_markers()
