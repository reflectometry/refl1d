"""
Layer interactor.
"""
from __future__ import division
from math import log
import numpy
from ..mystic.parameter import Parameter
from ..model import Slab
from ..polymer import PolymerBrush
from ..material import SLD, Vacuum, Material, Mixture, Compound
from .interactor import BaseInteractor
from .config import pick_radius
from .config import rho_color, rhoI_color, disable_color, profile_color
from .util import clip, setpar


class MaterialInteractor(BaseInteractor):
    def __init__(self, profile, material, range=(0,1)):
        super(MaterialInteractor,self).__init__(profile)
        self.material = material
        self.range = range
        
        if isinstance(material,SLD):
            enabled = True
            self._rho = material.rho
            self._rhoI = material.irho
            self._rho_scale = 1
            self._rhoI_scale = 1
        elif (hasattr(material,'density') 
              and isinstance(material.density,Parameter)):
            enabled = True
            rho,rhoI = material.sld(profile.experiment.probe)
            self._rho = material.density
            self._rhoI = material.density
            self._rho_scale = 1/ (rho / material.density.value)
            self._rhoI_scale = 1/ (rhoI / material.density.value)
        else:
            self._rho = self._rhoI = None
            enabled = False

        style = dict(linestyle='--',
                     linewidth  = 2,
                     pickradius = pick_radius,
                     zorder = 5,
                     )
        colors = rho_color,rhoI_color
        labels = "rho","rhoI"
        ax = profile.axes
        self.markers = [ax.plot([], [], 
                                label=material.name+" "+label, 
                                color=color if enabled else disabled_color,
                                **style)[0]
                        for color,label in zip(colors,labels)]
        if enabled:
            self.connect_markers(self.markers)
    
    def update_markers(self):
        z = self.profile.boundary[1:-1]
        n = self.profile.layer_num
        if n == 0:
            left,right = -20,0
        elif n >= len(z):
            left,right = z[-1],z[-1]+20
        else:
            delta = z[n] - z[n-1]
            left,right = self.range[0]*delta+z[n-1], self.range[1]*delta+z[n-1]

        rho,rhoI = self.material.sld(self.profile.experiment.probe)
        self.markers[0].set_xdata([left,right])
        self.markers[0].set_ydata([rho,rho])
        self.markers[1].set_xdata([left,right])
        self.markers[1].set_ydata([rhoI,rhoI])

    def save(self, ev):
        if self._rho is not None: self._rho_save = self._rho.value
        if self._rhoI is not None: self._rhoI_save = self._rhoI.value

    def restore(self, ev):
        if self._rho is not None: self._rho.value = self._rho_save
        if self._rhoI is not None: self._rhoI.value = self._rhoI_save

    def drag(self, ev):
        if ev.artist == self.markers[0]:
            setpar(self._rho, ev.ydata * self._rho_scale)
        else:
            setpar(self._rhoI, ev.ydata * self._rhoI_scale)

class SlabInteractor(BaseInteractor):
    def __init__(self, profile, layer):
        super(SlabInteractor,self).__init__(profile)
        self.material = MaterialInteractor(profile,layer.material)
    def update_markers(self):
        self.material.update_markers()
    def clear_markers(self):
        self.material.clear_markers()



#-----------------------------------------------------------------------
class PolymerBrushInteractor(BaseInteractor):
    """
    Interactor for tethered polymer model.
    """
    phi_scale = 0.4
    def __init__(self, profile, layer):
        super(PolymerBrushInteractor,self).__init__(profile)
        self.polymer = MaterialInteractor(profile,layer.polymer,
                                          range=(0,.3))
        self.solvent = MaterialInteractor(profile, layer.solvent,
                                          range=(0.7,1))
        self.layer = layer
        ax = profile.axes

        style = dict(linestyle = ':',
                     transform=profile.xcoords,
                     zorder=0,
                     color=profile_color,
                     visible=True,
                     )
        self.hprofile = ax.plot([],[], **style)[0]

        style = dict(linestyle='-',
                     transform=profile.xcoords,
                     zorder=4,
                     pickradius = pick_radius,
                     color = profile_color,
                     linewidth = 1,
                     visible=True,
                     )
        self.hphi = ax.plot( [], [], **style)[0]
        self.hbase = ax.plot([],[], **style)[0]
        self.hlength = ax.plot([],[], **style)[0]
        
        style = dict(marker='s',
                     transform=profile.xcoords,
                     zorder=5,
                     pickradius = pick_radius,
                     color = profile_color,
                     alpha = 0.5,
                     markersize = 5,
                     visible=True,
                     )
        self.hpower = ax.plot([],[], **style)[0]
        self.hsigma = ax.plot([],[], **style)[0]
        #print "xcoords",id(profile.xcoords),"trans",id(self.hbase.get_transform())

        self.markers = [self.hphi,self.hbase,self.hlength,
                        self.hpower,self.hsigma]
        self.parameters = [layer.base_vf,layer.base,layer.length,
                           layer.power,layer.sigma]
        self.connect_markers(self.markers)

    def clear_markers(self):
        super(PolymerBrushInteractor,self).clear_markers()
        self.polymer.clear_markers()
        self.solvent.clear_markers()
        self.hprofile.remove()

    def update_markers(self):
        """
        Draw the widgets in their new positions.
        """
        self.polymer.update_markers()
        self.solvent.update_markers()

        n = self.profile.layer_num
        left,right = self.profile.boundary[n:n+2]
        layer = self.layer

        z = numpy.linspace(0,layer.thickness.value,200)
        vf = layer.profile(z)
        self.hprofile.set_xdata(z+left)
        self.hprofile.set_ydata(vf*self.phi_scale)

        # at z midway between l1 and l2
        #     y = phi * 0.75**p
        phi = layer.base_vf.value*self.phi_scale/100
        phi_power = phi*0.75**layer.power.value
        L0 = left + layer.base.value
        L1 = L0 + layer.length.value
        Lpower = (L0+L1)/2
        Lsigma = L1 + layer.sigma.value
        
        #print "L0,L1,Lpower,Lsigma",L0,L1,Lpower,Lsigma,phi,phi_power
        self.hphi.set_xdata([left, L0])
        self.hphi.set_ydata([phi, phi])
        self.hbase.set_xdata([L0, L0])
        self.hbase.set_ydata([0, phi])
        self.hlength.set_xdata([L1, L1])
        self.hlength.set_ydata([0, phi])
        self.hpower.set_xdata([Lpower])
        self.hpower.set_ydata([phi_power])
        self.hsigma.set_xdata([Lsigma])
        self.hsigma.set_ydata([0])

    def drag(self, ev):
        """
        Update the model with the new widget position.
        """
        map = dict(zip(self.markers,self.parameters))
        par = map[ev.artist]
        n = self.profile.layer_num
        left,right = self.profile.boundary[n:n+2]
        if ev.artist == self.hphi:
            #print ev.ydata, self.profile.xcoords.inverted().transform([(ev.x,ev.y)])[0][1]
            setpar(par, ev.ydata*100/self.phi_scale)
            #print "phi",par.value,ev.ydata*100/self.phi_scale
        elif ev.artist == self.hbase:
            offset = left
            setpar(par, ev.xdata-offset)
        elif ev.artist == self.hlength:
            offset = left + self.layer.base.value
            setpar(par, ev.xdata-offset)
        elif ev.artist == self.hsigma:
            offset = left + self.layer.base.value + self.layer.length.value
            setpar(par, ev.xdata-offset)
        elif ev.artist == self.hpower:
            phi = self.layer.base_vf.value/100*self.phi_scale
            pow = log(ev.ydata/phi)/log(0.75) if ev.ydata > 0 else 100  
            setpar(par, pow)


    def save(self, ev):
        """
        Save the current state of the model represented by the widget.
        """
        map = dict(zip(self.markers,self.parameters))
        par = map[ev.artist]
        self._save_par = par
        self._save_val = par.value

    def restore(self, ev):
        """
        Restore the widget and model to the saved state.
        """
        self._save_par.value = self._save_val


#-----------------------------------------------------------------------
class SplineLayerInteractor(BaseInteractor):
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
class NoInteractor(BaseInteractor):
    """
    Null Interactor for undefined layers.
    """
    # TODO: turn layer.parameters() into sliders

# ======================== LayerInteractor factory ====================
# Associate layers with layer interactors through function
#     interactor(profile,layer)
# New layer interactors can be registered with
#     make_interactor[layer_class] = interactor_class
# =====================================================================
class _LayerInteractorFactory:
    """
    Given a layer, find the associated interactor.
    """
    def __init__(self):
        self._registry = {}
    def __setitem__(self, layer_class, interactor_class):
        self._registry[layer_class] = interactor_class
    def __call__(self, profile, layer):
        constructor = self._registry.get(layer.__class__, NoInteractor)
        return constructor(profile, layer)

make_interactor = _LayerInteractorFactory()
make_interactor[Slab] = SlabInteractor
make_interactor[PolymerBrush] = PolymerBrushInteractor
