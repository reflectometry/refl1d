"""
Reflectometry thickness interactor.
"""

from .config import interface_color, pick_radius
from .interactor import BaseInteractor

def sample_boundaries(experiment):
    boundaries = [0]
    offset = 0
    for L in experiment.sample[1:-1]:
        dx = L.thickness.value
        offset += dx
        boundaries.append(offset)
    return boundaries

def sample_labels(experiment):
    return [str(L) for L in experiment.sample]
    

class ThicknessInteractor(BaseInteractor):
    """
    Control the size of the layers.
    """
    def __init__(self,
                 profile,
                 axes,
                 color=interface_color
                 ):
        BaseInteractor.__init__(self, profile, axes, color=color)
        self.markers  = []
        self.textmark = []
        self._show_labels = True
        self._show_boundaries = True
        self.reset_layers()        

    def _offsets(self):
        boundaries = sample_boundaries(self.profile.experiment)
        left = -20
        middle = [(a+b)/2. for a,b in zip(boundaries[:-1],boundaries[1:])]
        right = boundaries[-1] + 20
        labels = [left]+middle+[right]
        return boundaries, labels
        
    def reset_layers(self):
        """
        Reset all markers.
        """
        self.clear_markers()
        ax = self.profile.ax

        boundaries, label_offsets = self._offsets()
        labels = sample_labels(self.profile.experiment)
                
        # Add bars
        style = dict(linewidth=1, linestyle='-',
                     color=interface_color, alpha=0.5,
                     pickradius=pick_radius,
                     visible=self._show_boundaries,
                     ) 
        self.markers = [ax.axvline(x=z, **style) for z in boundaries]

        self.markers[0].set(linestyle=':')
        self.connect_markers(self.markers[1:])

        # Add labels
        style = dict(transform=self.xcoords,
                     ha='left', va='bottom',
                     rotation=30, fontsize='small',
                     visible=self._show_labels,
                     )
                     
        self.textmark = [ax.text(z,1,s,**style)
                         for z,s in zip(label_offsets,labels)]

    def refresh(self):
        """
        Refreah all markers.

        Also we clear up all the connects with the markers
        """
        if  self.markers:
            self.base.connect.clear(*self.markers)

        self.reset_layers()

    def clear_markers(self):
        """
        Remove interfaces and layer names from the graph.
        """
        BaseInteractor.clear_markers(self)
        for h in self.textmark:
            h.remove()
        self.textmark = []


    def update(self):
        """
        Update the marker positions
        """
        boundaries, label_offsets = self._offsets()

        for z,h in zip(boundaries, self.markers):
            h.set_xdata([z,z])
        for z,h in zip(label_offsets, self.textmark):
            h.set_x(z)

    def save(self, ev):
        """
        Remember the depths for this layer and the next so that we
        can drag the boundary and restore it on Esc.
        """
        self.profile.freeze_axes()


    def restore(self):
        """
        Restore the depths for this layer and the next.
        """
        try:
            model = self.base.model
            model.depth[self._save_n] = self._save_d
            if self._save_n < model.numlayers:
                model.depth[self._save_n+1] = self._save_dnext
        except:
            pass


    def move(self, x, y, event):
        """
        Process move to a new position, making sure that the move is allowed.
        """
        layer_n = self._lookupLayerNumber(event)
        if layer_n == None:
            return

        model   = self.base.model
        lo = model.offset[layer_n]

        # Drag layer Depth
        min_depth = 1
        max_depth = 10000
        d = abs(x-lo)
        if  d <  min_depth:  # Too samll
            model.depth[ layer_n ] = min_depth
            return True

        if  d >= max_depth:  # Too Big
            model.depth[ layer_n ] = max_depth
            return True

        # update the depth in mode
        model.depth[ layer_n ] = d


    def moveend(self, event):
        self.base.thaw_axes()


    def showValue(self, event):
        """
        Show the depth Value
        """
        n = self._lookupLayerNumber(event)
        if n == None:
            return

        # Do we need save it?
        self._save_depth_n = n

        self.infopanel.updateNLayer( n )
        self.infopanel.showDepthValue(  )


    def setValue(self, event):
        """
        Set the depth Value

        First call move(), so we can directly use the updated data(depth).
        """
        n = self._lookupLayerNumber(event)
        if  n == None:
            return

        self.infopanel.updateNLayer( n )
        self.infopanel.updateDepthValue( self.base.model.depth[n] )
