"""
interface  interactor.
"""

from .config import roughness_color, pick_radius
from .interactor import BaseInteractor

class InterfaceInteractor(BaseInteractor):
    """
    Control the roughness of the layers.
    """
    def __init__(self,
                 base,
                 axes,
                 color=roughness_color
                 ):
        BaseInteractor.__init__(self, base, axes, color=color)

        self.layernum = 0
        self.axes     = axes

        # markers for roughness
        self.mrough=[self.axes.plot([0],[0.05],
                                    linestyle  = '',
                                    transform  = self.xcoords,
                                    marker     = 's', #square
                                    markersize = 7,
                                    color      = self.color,
                                    alpha      = 0.6,
                                    pickradius = pick_radius,
                                    label      = label,
                                    zorder     = 8, #Prefer this to other lines
                                    visible    = False)[0]
                           for label in ('left rough','right rough')
                     ]

        # lines for roughness
        self.lrough=[self.axes.plot([0,0],[0.05,0.05],
                                    transform = self.xcoords,
                                    linestyle = '-',
                                    marker    = '',
                                    color     = self.color,
                                    visible   = False)[0]
                           for label in ('left lrough','right lrough')
                     ]

        self.markers = self.mrough
        self.connect_markers(self.markers)


    def set_layer(self, n):
        """
        Set the layer number
        """
        self.layernum = n
        self.update()


    def update(self):
        """
        Draw the new roughness on the graph.
        """
        return
        model = self.base.model
        n     = self.layernum

        showLeft = (n>0 and n<=model.numlayers+1)
        self.mrough[0].set( visible = showLeft )
        self.lrough[0].set( visible = showLeft )
        if n > 0:
            self.mrough[0].set(xdata=[model.offset[n]+model.rough[n-1]])
            self.lrough[0].set(xdata=[model.offset[n],
                                      model.offset[n]+model.rough[n-1]])

        showRight = (n<=model.numlayers and n>=0)
        self.mrough[1].set( visible = showRight )
        self.lrough[1].set( visible = showRight )
        if n <= model.numlayers:
            self.mrough[1].set(xdata=[model.offset[n+1]-model.rough[n]])
            self.lrough[1].set(xdata=[model.offset[n+1],
                                      model.offset[n+1]-model.rough[n]])


    def clear(self):
        """
        clear roughness( line and marker) on the graph.
        """
        for line in self.lrough:
            line.remove()
        self.clear_markers()


    def save(self, event):
        """
        Remember the roughness for this layer and the next so that we
        can restore on Esc.
        """
        model = self.base.model
        self._save_n = self.mrough.index(event.artist)

        self._save_v = model.rough[self.layernum + self._save_n-1]

        # Freeze the x axes
        self.base.freeze_axes()


    def moveend(self, event):
        self.base.thaw_axes()


    def restore(self):
        """
        Restore the roughness for this layer.
        """
        try:
            model = self.base.model
            model.rough[self.layernum + self._save_n] = self._save_v
        except:
            pass


    def GetMaxRough(self, n):
        """
        Get the max rough for layer n
        """
        model = self.base.model
        if n == 0 :
            # You can change this number to show the depth in incident layer
            # is infinite
            return 1000.0
        else:
            return model.depth[n]/model.max_rough


    def move(self, x, y, event):
        """
        Process move to a new position, making sure that the move is allowed.
        """
        model = self.base.model
        n     = self.layernum

        self._save_n = self.mrough.index(event.artist)


        if self._save_n == 0:  # Left

            v = x - model.offset[n]
            if  v < 0:
                v = 0

            if  n >= 1 and  v > self.GetMaxRough(n-1):
                v = self.GetMaxRough(n-1)

            if  n <= model.numlayers and v > self.GetMaxRough(n):
                v = self.GetMaxRough(n)

            model.rough[n-1] = v

        else:  # Right

            v = model.offset[n+1] - x
            if  v < 0:
                v = 0

            if  n > 0 and v > self.GetMaxRough(n):
                v = self.GetMaxRough(n)

            if  n < model.numlayers and v > self.GetMaxRough(n+1):
                v = self.GetMaxRough(n+1)

            model.rough[n] = v


    def getRough(self, n):
        """
        Get the rough for layer n
        """
        try:
            val = self.base.model.rough[n]
        except:
            val = None
        return val


    def setValue(self, event):
        """
        Set the rough

        First call move(), so we can directly use the updated data(rough).
        """
        n     = self.layernum
        idx   = self.mrough.index(event.artist)

        if idx==1:
            val = self.getRough(n)
        else:
            val = self.getRough(n-1)

        if  val == None:
            return

        self.infopanel.updateNLayer( n )

        if  idx==1:  self.infopanel.updateRRoughValue( val )
        else:        self.infopanel.updateLRoughValue( val )


    def showValue(self, event):
        """
        Show the rough
        """
        n   = self.model.find( event.xdata )
        idx = self.mrough.index(event.artist)

        if idx ==1:  self._save_rough_n = n
        else:        self._save_rough_n = n-1

        self.infopanel.updateNLayer(  n )

        if  idx == 1:  self.infopanel.showRRoughValue( )
        else:          self.infopanel.showLRoughValue( )
