"""
Reflectometry thickness interactor.
"""

from .config import thickness_color, pick_radius
from .interactor import BaseInteractor
from .util import clip


class ThicknessInteractor(BaseInteractor):
    """
    Control the size of the layers.
    """

    def __init__(self, profile):
        BaseInteractor.__init__(self, profile)
        self.markers = []
        self.textmark = []
        self._show_labels = True
        self._show_boundaries = True

    def label_offsets(self):
        z = self.profile.boundary[1:-1]
        left = -20
        middle = [(a + b) / 2.0 for a, b in zip(z[:-1], z[1:])]
        right = z[-1] + 20
        return [left] + middle + [right]

    def reset_markers(self):
        """
        Reset all markers.
        """
        self.clear_markers()
        ax = self.profile.axes

        # Add bars
        style = dict(
            linewidth=1,
            linestyle="-",
            color=thickness_color,
            alpha=0.75,
            pickradius=pick_radius,
            visible=self._show_boundaries,
        )
        self.markers = [ax.axvline(x=z, **style) for z in self.profile.boundary[1:-1]]

        fittable = [self.profile.sample_layer(idx).thickness.fittable for idx, _ in enumerate(self.markers)]
        fittable[0] = False  # First interface is not fittable
        for f, m in zip(fittable, self.markers):
            if not f:
                m.set(linestyle="--", linewidth=1.25)
        self.connect_markers(m for f, m in zip(fittable, self.markers) if f)

        # Add labels
        offsets = self.label_offsets()
        labels = self.profile.sample_labels()
        style = dict(
            transform=self.profile.xcoords,
            ha="left",
            va="bottom",
            rotation=30,
            fontsize="small",
            visible=self._show_labels,
        )
        self.textmark = [ax.text(z, 1, s, **style) for z, s in zip(offsets, labels)]

    def clear_markers(self):
        """
        Remove interfaces and layer names from the graph.
        """
        BaseInteractor.clear_markers(self)
        for h in self.textmark:
            h.remove()
        self.textmark = []

    def update_markers(self):
        """
        Update the marker positions
        """
        label_offsets = self.label_offsets()

        for z, h in zip(self.profile.boundary[1:-1], self.markers):
            h.set_xdata([z, z])
        for z, h in zip(label_offsets, self.textmark):
            h.set_x(z)

    def save(self, ev):
        idx = self.markers.index(ev.artist)
        curr = self.profile.sample_layer(idx).thickness
        next = self.profile.sample_layer(idx + 1).thickness if idx < len(self.markers) - 1 else None
        if not curr.fittable:
            curr = None
        if next is not None and not next.fittable:
            next = None

        self._idx = idx
        self._curr = curr
        self._next = next
        if curr is not None:
            self._curr_saved = curr.value
        if next is not None:
            self._next_saved = next.value

    def restore(self, ev):
        self._curr.value = self._curr_saved
        if self._next is not None:
            self._next.value = self._next_saved

    def drag(self, event):
        """
        Process move to a new position, making sure that the move is allowed.
        """
        if self._curr is None:
            return

        prev_offset = self.profile.boundary[self._idx]
        next_offset = self.profile.boundary[self._idx + 2]

        # Limit the position according to parameter limits
        if event.shift and self._next is not None:
            # shifting interface between two layers
            lo_curr, hi_curr = self._curr.prior.limits
            lo_next, hi_next = self._next.prior.limits
            lo = min(lo_curr + prev_offset, next_offset - hi_next)
            hi = min(hi_curr + prev_offset, next_offset - lo_next)
        else:
            # resizing a layer
            lo_curr, hi_curr = self._curr.prior.limits
            lo = lo_curr + prev_offset
            hi = hi_curr + prev_offset
        x = clip(event.xdata, lo, hi)

        # Set the current and next value based on offsets.  We set the next
        # in case the user is shifting the interface between two layers
        # instead of resizing the current layer.  We set it even when not
        # shifting in case the previous action was to shift, and the current
        # action triggers a restore.  We set the next before the current
        # so that if the two parameters are tied, the current takes
        # precedence.
        if self._next is not None:
            if event.shift:
                self._next.value = next_offset - x
            else:
                self._next.value = self._next_saved
        self._curr.value = x - prev_offset

        # Adjust x_offset such that the layer boundary will stay under the
        # curesor even if previous layers grow or shrink as a result of being
        # tied to the current layer thickness.
        self.profile._find_layer_boundaries()
        self.profile.x_offset -= self.profile.boundary[self._idx] - prev_offset
