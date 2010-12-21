"""
Reflectometry thickness interactor.
"""
from numpy import inf

from .config import thickness_color, pick_radius
from .interactor import BaseInteractor
from .util import clip

class ThicknessInteractor(BaseInteractor):
    """
    Control the size of the layers.
    """
    def __init__(self, profile):
        BaseInteractor.__init__(self, profile)
        self.markers  = []
        self.textmark = []
        self._show_labels = True
        self._show_boundaries = True

    def label_offsets(self):
        z = self.profile.boundary[1:-1]
        left = -20
        middle = [(a+b)/2. for a,b in zip(z[:-1],z[1:])]
        right = z[-1] + 20
        return [left]+middle+[right]
        
    def reset_markers(self):
        """
        Reset all markers.
        """
        self.clear_markers()
        ax = self.profile.axes
                
        # Add bars
        style = dict(linewidth=1, linestyle='-',
                     color=thickness_color, alpha=0.5,
                     pickradius=pick_radius,
                     visible=self._show_boundaries,
                     ) 
        self.markers = [ax.axvline(x=z, **style) 
                        for z in self.profile.boundary[1:-1]]

        self.markers[0].set(linestyle=':')
        self.connect_markers(self.markers[1:])

        # Add labels
        offsets = self.label_offsets()
        labels = self.profile.sample_labels()
        style = dict(transform=self.profile.xcoords,
                     ha='left', va='bottom',
                     rotation=30, fontsize='small',
                     visible=self._show_labels,
                     )
        self.textmark = [ax.text(z,1,s,**style)
                         for z,s in zip(offsets,labels)]

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

        for z,h in zip(self.profile.boundary[1:-1], self.markers):
            h.set_xdata([z,z])
        for z,h in zip(label_offsets, self.textmark):
            h.set_x(z)

    def save(self, ev):
        idx = self.markers.index(ev.artist)
        self._curr = self.profile.sample_layer(idx).thickness
        self._curr_saved = self._curr.value
        self._prev_offset = self.profile.boundary[idx]
        self._next_offset = self.profile.boundary[idx+2]
        if idx<len(self.markers)-1: # last
            self._next = self.profile.sample_layer(idx+1).thickness
            self._next_saved = self._next.value
        else:
            self._next = None
    
    def restore(self, ev):
        self._curr.value = self._curr_saved
        if self._next is not None:
            self._next.value = self._next_saved

    def drag(self, event):
        """
        Process move to a new position, making sure that the move is allowed.
        """
        if event.shift:
            # move boundary between layers
            lo_curr,hi_curr = self._curr.bounds.limits
            lo_next,hi_next = self._next.bounds.limits
            lo = min(lo_curr+self._prev_offset,
                     self._next_offset-hi_next)
            hi = max(hi_curr+self._prev_offset,
                     self._next_offset-lo_next)
        else:
            lo_curr,hi_curr = self._curr.bounds.limits
            lo = lo_curr+self._prev_offset
            hi = hi_curr+self._prev_offset

        x = clip(event.xdata, lo, hi)
        self._curr.value = x - self._prev_offset
        if self._next is not None:
            self._next.value = (self._next_offset-x if event.shift 
                                else self._next_saved)

    def drag_start(self, ev):
        """
        Remember the depths for this layer and the next so that we
        can drag the boundary and restore it on Esc.
        """
        self.profile.freeze_axes()
        
    def drag_cancel(self, event):
        """
        Restore the depths for this layer and the next.
        """
        self.profile.thaw_axes()

    def drag_done(self, event):
        self.profile.thaw_axes()
