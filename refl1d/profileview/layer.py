"""
Layer interactor.
"""
from __future__ import division
from math import log
import numpy
from ..mystic.parameter import Parameter
from ..material import SLD, Vacuum, Material, Mixture, Compound
from .interactor import BaseInteractor
from .config import pick_radius
from .config import rho_color, rhoI_color, disabled_color, profile_color
from .config import vf_scale
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
        self.markers[0].set_data( (left,right), (rho,rho) )
        self.markers[1].set_data( (left,right), (rhoI,rhoI) )

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
        self.hprofile.set_data(z+left, vf*vf_scale)

        # at z midway between l1 and l2
        #     y = phi * 0.75**p
        phi = layer.base_vf.value*vf_scale/100
        phi_power = phi*0.75**layer.power.value
        L0 = left + layer.base.value
        L1 = L0 + layer.length.value
        Lpower = (L0+L1)/2
        Lsigma = L1 + layer.sigma.value
        
        #print "L0,L1,Lpower,Lsigma",L0,L1,Lpower,Lsigma,phi,phi_power
        self.hphi.set_data( (left,L0), (phi,phi) )
        self.hbase.set_data( (L0,L0),  (0,phi) )
        self.hlength.set_data( (L1,L1), (0, phi) )
        self.hpower.set_data( (Lpower,), (phi_power,) )
        self.hsigma.set_data( (Lsigma,), (0,) )

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
            setpar(par, ev.ydata*100/vf_scale)
            #print "phi",par.value,ev.ydata*100/vf_scale
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
            phi = self.layer.base_vf.value/100*vf_scale
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
class FreeInterfaceInteractor(BaseInteractor):
    """
    Interactor for freeform interfaces using monotonic splines.
    """
    def __init__(self, profile, layer):
        super(FreeInterfaceInteractor,self).__init__(profile)
        self.polymer = MaterialInteractor(profile,layer.below,
                                          range=(0,.3))
        self.solvent = MaterialInteractor(profile, layer.above,
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

        style = dict(marker='o',
                     transform=profile.xcoords,
                     zorder=5,
                     pickradius = pick_radius,
                     color = profile_color,
                     alpha = 0.5,
                     markersize = 5,
                     visible=True,
                     )
        self.markers = [ax.plot([],[], **style)[0]
                        for _ in layer.dp]
        self.connect_markers(self.markers)

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
        self.hprofile.set_ydata(vf*vf_scale)

        z = cumsum([v.value for v in layer.dz])
        p = cumsum([v.value for v in layer.dp])
        if p[-1] == 0: p[-1] = 1
        p *= 1/p[-1]
        z *= thickness/z[-1]
        for h,zi,pi in zip(markers,z,p):
            h.set_data( (zi,), (pi,) )            
        
        profile = monospline(z, p, Pz)
        return profile


    def drag(self, evt):
        """
        Update the model with the new widget position.
        """

    def save(self, evt):
        """
        Save the current state of the model represented by the widget.
        """

    def restore(self):
        """
        Restore the widget and model to the saved state.
        """

# -------------------------------------------------------------------
class NoInteractor(BaseInteractor):
    """
    Null Interactor for undefined layers.
    """
    # TODO: turn layer.parameters() into sliders
