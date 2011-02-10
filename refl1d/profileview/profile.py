"""
Reflectometry profile interactor.
"""

import wx
import numpy
from numpy import inf
from wx.lib.pubsub import Publisher as pub
from .binder import BindArtist, pixel_to_data
from .util import twinx
from .config import rho_color, rhoI_color, rhoM_color, thetaM_color
from .config import layer_hysteresis

from . import registry
from .interactor import BaseInteractor, safecall
from .thickness import ThicknessInteractor
from .interface import InterfaceInteractor

from matplotlib import transforms
try:
    # matplotlib 0.98
    blend_xy = transforms.blended_transform_factory
except:
    # CRUFT: matplotlib 0.91
    blend_xy = transforms.blend_xy_sep_transform

# ================== Main profile interactor =====================
class ProfileInteractor(object):
    """
    Reflectometry profile editor
    """
    _debug = False
    def __init__(self, axes, experiment, listener):
        self.axes = axes
        self.xcoords = blend_xy(axes.transData, axes.transAxes)
        self.listener = listener
        self.experiment = experiment
        self.magnetic = experiment.sample.magnetic

        # Theta needs a separate axis, we put these two axes into a figure
        if self.magnetic:
            self.axes2 = twinx( self.axes )
        else:
            self.axes2 = None

        self.axes.set_xlabel( r'$\rm{z}\ (\AA)$' )
        if self.magnetic:
            self.axes.set_ylabel( r'$\rm{Density}\ \times 10^{-6}\ \ \rho,\  \rho_i,\  \rho_M$')
        else:
            self.axes.set_ylabel(r'$\rm{Density}\ \times 10^{-6}\ \ \rho,\  \rho_i$')

        if self.magnetic:
            self.axes2.set_ylabel(r'$\rm{Magnetic\ Angle\ (\ ^\circ)}$')

        # TODO: the connect mechanism needs to be owned by the canvas rather
        # than the axes --- cannot have multiple profiles on the same canvas
        # until connect is in the right place.
        self.connect = BindArtist( axes.figure )
        self.connect.clearall()
        self.connect('motion', axes, self.onMotion )
        self.connect('click', axes.figure, self.onContext)

        # Add some plots
        [self.hrho ] = axes.plot([],[],'-',color=rho_color,label=r'$\rho$')
        [self.hrhoI] = axes.plot([],[],'-',color=rhoI_color,label=r'$\rho_i$')
        if self.magnetic:
            [self.hrhoM  ] = self.axes.plot(  [], [], '-', color=rhoM_color,
                                            label = r'$\rho_M$')
            [self.hthetaM] = self.axes2.plot( [], [], '-', color=thetaM_color,
                                            label = r'$\theta_M$')

        # Show Legend ?
        if self.magnetic:
            self.hlegend = self.axes.legend(
                           (self.hrho, self.hrhoI, self.hrhoM, self.hthetaM),
                           #('SLD','Absorption','Mag. SLD','Mag. angle'),
                           (r'$\rho$', r'$\rho_i$', r'$\rho_M$', r'$\theta_M$'),
                           loc = (0.85,0.5)
                           #loc='upper right'
                           )
        else:
            self.hlegend = self.axes.legend( (self.hrho, self.hrhoI),
                                           (r'$\rho$', r'$\rho_i$'),
                                           loc = (0.85,0.5)
                                           #loc='upper right'
                                           )
        self.hlegend.get_frame().set( alpha=0.2, facecolor='yellow' )


        # Set interactors
        self.x_offset = 0
        self._find_layer_boundaries()
        self.layer_num = None
        self.thickness_interactor = ThicknessInteractor(self)
        self.interface_interactor = InterfaceInteractor(self)
        self.layer_interactor = BaseInteractor(self)
        self.thickness_interactor.reset_markers()
        self.interface_interactor.set_layer()

        # update the figure
        self.axes_frozen = False
        self.update()

    def onMotion(self, event):
        """Respond to motion events by changing the active layer."""
        # Force data coordinates for the mouse position
        transform = self.axes.transData
        x,_ = pixel_to_data(transform, event.x, event.y)

        # Check if mouse is still in the same layer
        if self.layer_num is not None:
            left,right = self.boundary[self.layer_num:self.layer_num+2]
            # Quick rejection if it is in the layer
            if left < x < right:
                #print "quick reject", event.xdata, self.layer_start, self.layer_end
                return False
            # If it is within 5 pixels of the layer, keep to the same layer
            xlo,_ = pixel_to_data(transform, event.x+layer_hysteresis, event.y)
            xhi,_ = pixel_to_data(transform, event.x-layer_hysteresis, event.y)
            if left < xlo and xhi < right:
                #print "hysteresis", event.x, xlo, xhi, self.layer_start, self.layer_end
                return False

        # If layer changed, then replace the markers
        self.set_layer(x)
        return False


    def onContext(self, ev):
        """Context menu (eventually ...)."""
        return False

    # === Sample information ===
    def find(self, z):
        return self.experiment.sample.find(z-self.x_offset)

    def _find_layer_boundaries(self):
        offset = self.x_offset
        boundary = [-inf, offset]
        for L in self.experiment.sample[1:-1]:
            dx = L.thickness.value
            offset += dx
            boundary.append(offset)
        boundary.append(inf)
        self.boundary = numpy.asarray(boundary)

    def sample_labels(self):
        return [str(L) for L in self.experiment.sample]

    def sample_layer(self, n):
        return self.experiment.sample[n]

    @safecall
    def set_layer(self, z):
        """Make layer containing z the active layer."""
        self.layer_num = numpy.searchsorted(self.boundary, z)-1
        layer = self.sample_layer(self.layer_num)

        # Reset the interface markers
        self.interface_interactor.set_layer()

        # Reset the layer markers
        self.layer_interactor.clear_markers()
        self.layer_interactor = registry.interactor(self, layer)

        # Update the marker positions
        self.update_markers()

        # Trigger redraw
        self.draw_idle()

    def update_markers(self):
        self._find_layer_boundaries()
        self.thickness_interactor.update_markers()
        self.interface_interactor.update_markers()
        self.layer_interactor.update_markers()

    def update_profile(self):
        self.experiment.update()
        if self.magnetic:
            z,rho,rhoI,rhoM,thetaM = self.experiment.smooth_profile()
            self.hrhoM.set_data(z,rhoM)
            self.hthetaM.set_data(z,thetaM)
        else:
            z,rho,rhoI = self.experiment.smooth_profile()
        self.hrho.set_data(z+self.x_offset,rho)
        self.hrhoI.set_data(z+self.x_offset,rhoI)

    def reset_limits(self):

        # Compute automatic y limits
        # Note: theta limits are on ax2

        # TODO: the markers themselves may fall outside the graph.
        # May need to walk sample to find the position of the markers
        # when setting the limits.  Alternatively, we could just ask
        # where the markers are.
        z = self.hrho.get_xdata()
        rho = self.hrho.get_ydata()
        rhoI = self.hrhoI.get_ydata()
        self.axes.set_xlim(z[0],z[-1])

        if self.magnetic:
            rhoM = self.hrhoM.get_ydata()
            thetaM = self.hthetaM.get_ydata()
            lo = min( rho.min(), rhoI.min(), rhoM.min() )
            hi = max( rho.max(), rhoI.max(), rhoM.max() )
            fluff = 0.05*(hi-lo)
            self.axes.set_ylim(lo-fluff, hi+fluff)

            lo, hi = thetaM.min(), thetaM.max()
            fluff = 0.05*(hi-lo)
            self.axes2.set_ylim(lo-fluff, hi+fluff)

        else:
            lo = min( rho.min(), rhoI.min() )
            hi = max( rho.max(), rhoI.max() )
            fluff = 0.05*(hi-lo)
            self.axes.set_ylim(lo-fluff, hi+fluff)

    def redraw(self):
        self.update_markers()
        self.update_profile()
        self.draw_now()
    
    def update(self):
        """
        Respond to changes in the model by recalculating the profiles and
        resetting the widgets.
        """
        # We are done the manipulation; let the model send its update signal
        # to whomever is listening.
        #self.listener.signal('update',self)
        self.redraw()
        self.delayed_signal()
        #self.delayed_profile()

    def delayed_profile(self):
        try: self._delayed.Restart(50)
        except: self._delayed = wx.FutureCall(50, lambda:(self.update_profile(),self.draw_now()))

    def _signal(self):
        try:
            pub.sendMessage("update_parameters")
        except:
            print 'error in message sending'
            raise
        self.listener.signal('update',self)
    def delayed_signal(self):
        try: self._delayed_signal.Restart(50)
        except: self._delayed_signal = wx.FutureCall(50, self._signal)

    def freeze_axes(self):
        self.axes_frozen = True

    def thaw_axes(self):
        self.axes_frozen = False

    def draw_now(self):
        #try: del self._delayed
        #except: pass
        self.axes.figure.canvas.draw()
    def draw_idle(self):
        """Set the limits and tell the canvas to render itself."""
        #print "draw when idle"
        self.axes.figure.canvas.draw_idle()
