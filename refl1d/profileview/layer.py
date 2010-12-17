"""
Layer interactor.
"""

import numpy
from ..model import Slab
from .interactor import BaseInteractor
from .config import pick_radius


# ===================== Layer interactors ========================
# GUI starts here
# ================================================================
class LayerInteractor(BaseInteractor):
    """
    Flat layer interactor.
    """
    def __init__(self,
                 profile,
                 axes,
                 layer,
                 color='black'
                 ):
        super(LayerInteractor,self).__init__(profile, axes, color=color)
        self.layer   = layer
        self.markers = []

    def _lookupIndex( self, event ):

        try:    idx = self.layerMarker.index(event.artist)
        except: idx = None
        return idx


    def  updateRhoValue(self, name, event):
        """ Update the rho layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.updateRhoValue(  event.ydata, idx=idx )


    def  updateMuValue(self, name, event):
        """ Update the mu layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.updateMuValue(  event.ydata, idx=idx )


    def  updatePhiValue(self, name, event):
        """ Update the phi layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.updatePhiValue(  event.ydata, idx=idx )


    def updateThetaValue( self, name, event):

        """ Update the theta layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.updateThetaValue(  event.ydata, idx=idx )


    def IsFlatLayer( self, label):
        # Obtain Artist name
        name = label.split("[")

        if len(name) <= 1:
            return  True
        else:
            return  False


    def _GetBestCurrLayerNum(self, event):
        idx = self._lookupIndex( event )
        if   idx==0:                         return self._save_depth_n +1
        elif idx == self.getMarkerSize()-1:  return self._save_depth_n
        else:                                return self._save_n


    def getBestCurrLayerNum(self, event):
        if  self.IsFlatLayer(  event.artist.get_label() ):
            return self._save_n
        else:
            return self._GetBestCurrLayerNum(event)


    def setValue(self, event):
        """ Update the layer value """
        _pn = self.Artist2Name( event.artist.get_label() )

        self._curr_n = self.getBestCurrLayerNum(event)

        if len(_pn) > 0:

            if    _pn[:3] == "rho":   self.updateRhoValue(   _pn, event)
            elif  _pn[:2] == "mu":    self.updateMuValue(    _pn, event)
            elif  _pn[:3] == "phi":   self.updatePhiValue(   _pn, event)
            elif  _pn[:5] == "theta": self.updateThetaValue( _pn, event)
            else:
                raise ValueError("Invalid parmeter")


    def  showRhoValue(self, event):
        """ Show the rho layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.showRhoValue( idx=idx )


    def  showMuValue(self, event):
        """ show the mu layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.showMuValue( idx=idx )


    def  showPhiValue(self, event):
        """ show the phi layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.showPhiValue( idx=idx )


    def  showThetaValue( self, event):
        """ show the theta layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.showThetaValue( idx=idx )


    def showValue(self, event):
        """ Update the layer value """
        _pn = self.Artist2Name( event.artist.get_label() )

        self._curr_n = self.getBestCurrLayerNum(event)

        if len(_pn) > 0:

            if    _pn[:3] == "rho":   self.showRhoValue(  event)
            elif  _pn[:2] == "mu":    self.showMuValue(   event)
            elif  _pn[:3] == "phi":   self.showPhiValue(  event)
            elif  _pn[:5] == "theta": self.showThetaValue(event)
            else:
                raise ValueError("Invalid parmeter name")


    # ----------------------------------------------------------
    def get_Marker(self, i ):
        filled_markers =[ 'o', # '_draw_circle',
                          's', # '_draw_square',
                          'p', # '_draw_pentagon',
                          'd', # '_draw_thin_diamond',
                          'h', # '_draw_hexagon1',
                          '+', # '_draw_plus',
                          'x', # '_draw_x',
                          'D', # '_draw_diamond',
                          'H', # '_draw_hexagon2',
                          'v', #'_draw_triangle_down',
                          '^', # '_draw_triangle_up',
                          '<', # '_draw_triangle_left',
                          '>', # '_draw_triangle_right',
                          '1', # '_draw_tri_down',
                          '2', # '_draw_tri_up',
                          '3', # '_draw_tri_left',
                          '4', # '_draw_tri_right',
                        ]
        return filled_markers[ i%len(filled_markers) ]




# ---------------------------------------------------------------------
class FlatLayerInteractor(LayerInteractor):
    """
    Interactor for FlatLayer to handle flat slabs.
    """
    def set_layer(self, n, show=True):
        """
        Setup the widgets required to edit layer n.
        """
        self.layernum = n
        v  = self.layer._val
        ax = self.axes

        self.layerMarker = ax.plot( [], [],
                                    '--',
                                    label      = self.par,
                                    linewidth  = 2,
                                    color      = self.color,
                                    pickradius = pick_radius,
                                    zorder     = 5
                                    )[0]
        self.markers = [ self.layerMarker ]

        self.connect_markers(self.markers)

        self.update(show)


    def update(self, show=True):
        """
        Draw the widgets in their new positions.
        """
        model = self.base.model
        n = self.layernum
        x = [ model.offset[n], model.offset[n+self.layer.span] ]
        y = [self.layer._val]*2

        h = self.markers[0]
        h.set_data(x,y)
        h.set_visible(show)


    def move(self, x, y, evt):
        """
        Update the model with the new widget position.
        """
        self.layer._val = y


    def save(self, evt):
        """
        Save the current state of the model represented by the widget.
        """
        self._saved_v = self.layer._val


    def restore(self):
        """
        Restore the widget and model to the saved state.
        """
        self.layer._val = self._saved_v




# ------------------------------------------------------------------
class SlopeLayerInteractor(LayerInteractor):
    """
    Interactor for SlopeLayer to handle the line control points.

    For slope layer, we use 'p' marker
    """
    def getMarkerSize(self):
        return len(self.layerMarker)


    def set_layer(self, n):
        """
        Setup the widgets required to edit layer n.
        """
        self.layernum = n

        left, right = self.layer._val
        ax = self.axes

        self.layerMarker = [ ax.plot( [], [],
                       linestyle  = '',
                       markersize = 10,
                       label      = "%s[%d]"%(self.par,i),
                       linewidth  = 2,
                       color      = self.color,
                       pickradius = pick_radius,
                       zorder     = 3,
                       alpha      = 0.6,
                       marker     = 'p',
                       visible    = False
                       )[0]  for i in xrange( 2 ) ]

        slopeLine = ax.plot( [], [],
                       '--',
                       label      = 'slope::line::'+self.par,
                       linewidth  = 2,
                       color      = self.color,
                       pickradius = 0,
                       zorder     = 5,
                       visible = False
                       )[0]


        self.markers = [self.layerMarker[0], self.layerMarker[1]]

        self.connect_markers(self.markers)
        self.markers.append( slopeLine )

        self.update()



    def update(self):
        """
        Draw the widgets in their new positions.
        """
        model = self.base.model
        n = self.layernum

        # We shift 5 point to avoid overlapping with depth marker
        left_x  = [ model.offset[n]]
        right_x = [ model.offset[n+self.layer.span] ]

        left_y  = [self.layer._val[0] ]
        right_y = [self.layer._val[1] ]

        leftMarker  = self.markers[0]
        rightMarker = self.markers[1]
        lineMarker  = self.markers[2]

        leftMarker.set( visible=(n>0))
        lineMarker.set( visible=(n>0))
        rightMarker.set(visible=(n>0))

        m_x =  [ model.offset[n], model.offset[n+self.layer.span] ]
        m_y =  [self.layer._val[0],  self.layer._val[1] ]

        leftMarker.set_data( left_x,  left_y )
        rightMarker.set_data(right_x, right_y)
        lineMarker.set_data( m_x,     m_y    )



    def move(self, x, y, evt):
        """
        Update the model with the new widget position.
        """
        idx = self._lookupIndex( evt )

        if  idx != None :
            self.layer._val[ idx ] = y

        # Otherwise: Do Nothing.



    def save(self, evt):
        """
        Save the current state of the model represented by the widget.
        """
        self._saved_v = self.layer._val



    def restore(self):
        """
        Restore the widget and model to the saved state.
        """
        self.layer._val = self._saved_v




#-----------------------------------------------------------------------
class TetheredPolymerInteractor(LayerInteractor):
    """
    Interactor for TetheredPolymer to handle bspline control points.

    For TetheredPolymer layer, we use "circle" marker
    """
    def Artist2Name( self, label):
        # Obtain Artist name
        ValidParNames = ["mu", "rho", "theta", "phi"]
        name = label.split("_")[0].strip()

        if name in ValidParNames:
            return  label
        else:
            return  ""


    def setValue(self, event):
        """ Update the layer value """
        _pn = self.Artist2Name( event.artist.get_label() )

        self._curr_n = self.getBestCurrLayerNum(event)

        if len(_pn) > 0:

            if    _pn[:3] == "rho":   self.updateRhoValue(   _pn, event)
            elif  _pn[:2] == "mu":    self.updateMuValue(    _pn, event)
            elif  _pn[:3] == "phi":   self.updatePhiValue(   _pn, event)
            elif  _pn[:5] == "theta": self.updateThetaValue( _pn, event)
            else:
                raise ValueError("Invalid parmeter")


    def  updateRhoValue(self, name, event):
        """ Update the rho layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        #print idx, name
        if idx == 2:
            self.infopanel.updateRhoValue(  event.xdata,
                                            idx=idx,
                                            name=name )
            return
        self.infopanel.updateRhoValue(  event.ydata, idx=idx, name=name )


    def  updateMuValue(self, name, event):
        """ Update the mu layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.updateMuValue(  event.ydata, idx=idx, name=name )


    def  updatePhiValue(self, name, event):
        """ Update the phi layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.updatePhiValue(  event.ydata, idx=idx, name=name )


    def updateThetaValue( self, name, event):

        """ Update the theta layer """
        n = self._curr_n
        self.infopanel.updateNLayer( n )

        idx = self._lookupIndex( event )
        self.infopanel.updateThetaValue(  event.ydata, idx=idx, name=name )


    def getMarkerSize(self):
        return len(self.layerMarker)


    def set_layer(self, n):
        """
        Setup the widgets required to edit layer n.
        """
        self.layernum = n

        ax = self.axes

        polymerLine = ax.plot( [], [],
                               '--',
                               label      = "%s_polymerSLD"%self.par,
                               linewidth  = 2,
                               color      = self.color,
                               pickradius = pick_radius,
                               zorder     = 5,
                               visible = False
                               )[0]

        solventLine = ax.plot( [], [],
                               '--',
                               label      = "%s_solventSLD"%self.par,
                               linewidth  = 2,
                               color      = self.color,
                               pickradius = pick_radius,
                               zorder     = 5,
                               visible = False
                               )[0]

        L0Line = ax.axvline(x=0,
                            linewidth=2,
                            linestyle='--',
                            label="%s_L0"%self.par,
                            color=self.color,
                            alpha=0.5,
                            pickradius=pick_radius
                            )

        self.layerMarker = [ polymerLine, solventLine,  L0Line]

        self.markers = []
        for i in xrange( self.getMarkerSize() ):
            self.markers.append(self.layerMarker[i])

        self.connect_markers(self.markers)
        self.update()


    def update(self):
        """
        Draw the widgets in their new positions.
        """
        model = self.base.model
        n     = self.layernum

        left_x  = model.offset[n]
        right_x = model.offset[n+self.layer.span]
        span    = right_x - left_x

        L0 = self.layer._val[2]
        nv = 2

        if span*0.1 > L0:  Lshift = L0
        else:              Lshift = span*0.1

        control = [
                    [left_x, left_x+Lshift],
                    [right_x-span*0.1, right_x]
                  ]
        for i in xrange(nv):
            self.markers[i].set(visible=(n>0))

        # Spline line
        for i in xrange(nv):
            m_x = [ control[i][0],       control[i][1]      ]
            m_y = [ self.layer._val[i],  self.layer._val[i] ]
            self.markers[i].set_data(m_x, m_y)

        self.markers[2].set_xdata( [L0, L0])


    def move(self, x, y, evt):
        """
        Update the model with the new widget position.
        """
        idx = self._lookupIndex( evt )

        if idx != None :
            self.layer._val[idx] = y


    def save(self, evt):
        """
        Save the current state of the model represented by the widget.
        """
        self._saved_v = self.layer._val


    def restore(self):
        """
        Restore the widget and model to the saved state.
        """
        self.layer._val = self._saved_v



#-----------------------------------------------------------------------
class SplineLayerInteractor(LayerInteractor):
    """
    Interactor for SplineLayer to handle bspline control points.

    For spline layer, we use "circle" marker
    """
    def getMarkerSize(self):
        return len(self.layerMarker)


    def set_layer(self, n):
        """
        Setup the widgets required to edit layer n.
        """
        self.layernum = n

        ax = self.axes

        splineLines = [ ax.plot( [], [],
                               '--',
                               label      = 'slope::line::'+self.par,
                               linewidth  = 2,
                               color      = self.color,
                               pickradius = 0,
                               zorder     = 5,
                               visible = False
                               )[0]  
                        for i in xrange(len(self.layer._val)-1) ]

        self.layerMarker = [ax.plot( [], [],
                                 linestyle='',
                                 markersize = 10,
                                 label      = "%s[%d]"%(self.par,i),
                                 linewidth  = 2,
                                 color      = self.color,
                                 pickradius = pick_radius,
                                 zorder     = 3,
                                 alpha   = 0.6,
                                 marker  = 'o',
                                 visible = False
                                 )[0]  
                            for i in xrange( len(self.layer._val) ) ]


        # FIXME: use fast way to combine two lists into a single list
        self.markers = []
        for i in xrange( len(self.layerMarker) ):
            self.markers.append(self.layerMarker[i])

        self.connect_markers(self.markers)

        for i in xrange( len(splineLines) ):
            self.markers.append( splineLines[i] )

        self.update()


    def update(self):
        """
        Draw the widgets in their new positions.
        """
        model = self.base.model
        n = self.layernum

        left_x  = model.offset[n]
        right_x = model.offset[n+self.layer.span]
        span    = right_x - left_x

        nv = len( self.layer._val )
        control_z = numpy.arange(0.0, nv)/(nv-1.0)*span + left_x

        for i in xrange(nv*2-1):
            self.markers[i].set(visible=(n>0))

        # spline Markers
        for i in xrange(nv):
            self.markers[i].set_data(control_z[i], self.layer._val[i])

        # spline line
        for i in xrange(nv-1):
            m_x = [ control_z[i],        control_z[i+1]       ]
            m_y = [ self.layer._val[i],  self.layer._val[i+1] ]
            self.markers[i+nv].set_data(m_x, m_y)


    def move(self, x, y, evt):
        """
        Update the model with the new widget position.
        """
        idx = self._lookupIndex( evt )
        if  idx != None :
            self.layer._val[idx] = y


    def save(self, evt):
        """
        Save the current state of the model represented by the widget.
        """
        self._saved_v = self.layer._val


    def restore(self):
        """
        Restore the widget and model to the saved state.
        """
        self.layer._val = self._saved_v




# -------------------------------------------------------------------
class NoLayerInteractor(LayerInteractor):
    """
    Null Interactor for undefined layers.
    """
    def set_layer(self, n):
        pass

    def update(self):
        pass

    def move(self, x, y):
        pass

    def save(self):
        pass

    def restore(self):
        pass



# ======================== LayerInteractor factory ====================
# Associate layers with layer interactors through function
#     interactor(layer)
# =====================================================================
class _LayerInteractorFactory:
    """
    Given a layer, find the associated interactor.
    """
    def __init__(self):
        self.template = {Slab: FlatLayerInteractor
                         },
#                         SlopeLayer: SlopeLayerInteractor,
#                         SplineLayer: SplineLayerInteractor,
#                         TetheredPolymerLayer: TetheredPolymerInteractor,
#                         JoinLayer: NoLayerInteractor,
#                         NoLayer: NoLayerInteractor
#                         }
    def register(self, layer_class, interactor_class):
        self.template[layer_class] = interactor_class
    def __call__(self,
                 profile,
                 layer,
                 **kw
                 ):
        if layer.__class__ in self.template:
            return self.template[layer.__class__](profile, layer, **kw)
        else:
            return NoLayerInteractor(profile, layer, **kw)

make_interactor = _LayerInteractorFactory()
