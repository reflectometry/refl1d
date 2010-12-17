"""
Reflectometry profile interactor.
"""

from numpy import inf

from .binder import BindArtist, pixel_to_data
from .util import twinx
from .config import interface_color, disable_color, active_color, title_color, \
                 rho_color, rhoI_color, rhoM_color, thetaM_color
from .config import pick_radius

from .layer import make_interactor
from .thickness import ThicknessInteractor
from .interface import InterfaceInteractor

# ================== Main profile interactor =====================
class ProfileInteractor(object):
    """
    Reflectometry profile editor
    """
    def __init__(self,
                 ax,
                 experiment,
                 listener,
                 ):
        self.ax      = ax
        self.listener= listener
        self.experiment = experiment
        self.magnetic = experiment.sample.magnetic


        # Theta needs a separate axis, we put these two axes into a figure
        if self.magnetic:
            self.ax2 = twinx( self.ax )
        else:
            self.ax2 = None

        self.ax.set_xlabel( r'$\rm{z}\ (\AA)$' )
        if self.magnetic:
            self.ax.set_ylabel( r'$\rm{Density}\ \times 10^{-6}\ \ \rho,\  \rho_i,\  \rho_M$')
        else:
            self.ax.set_ylabel(r'$\rm{Density}\ \times 10^{-6}\ \ \rho,\  \rho_i$')

        if self.magnetic:
            self.ax2.set_ylabel(r'$\rm{Magnetic\ Angle\ (\ ^\circ)}$')

        # TODO: the connect mechanism needs to be owned by the canvas rather
        # than the axes --- cannot have multiple profiles on the same canvas
        # until connect is in the right place.
        self.connect = BindArtist( ax.figure )
        self.connect.clearall()
        self.connect('motion', ax,        self.onMotion )
        self.connect('click',  ax.figure, self.onContext)

        # Add interactor for layer thickness and interface
        self.thickness_interactor = ThicknessInteractor(self,ax)
        self.interface_interactor = InterfaceInteractor(self,ax)
        self.segment_interactor = None
        self.layer, self.layer_start, self.layer_end = None, 1, -1

        # Add some plots
        [self.hrho ] = ax.plot([],[],'-',color=rho_color,label=r'$\rho$')
        [self.hrhoI] = ax.plot([],[],'-',color=rhoI_color, label=r'$\rho_i$' )
        if self.magnetic:
            [self.hrhoM  ] = self.ax.plot(  [], [], '-', color=rhoM_color,
                                            label = r'$\rho_M$')
            [self.hthetaM] = self.ax2.plot( [], [], '-', color=thetaM_color,
                                            label = r'$\theta_M$')

        # Show Legend ?
        if self.magnetic:
            self.hlegend = self.ax.legend(
                           (self.hrho, self.hrhoI, self.hrhoM, self.hthetaM),
                           #('SLD','Absorption','Mag. SLD','Mag. angle'),
                           (r'$\rho$', r'$\rho_i$', r'$\rho_M$', r'$\theta_M$'),
                           loc = (0.85,0.5)
                           #loc='upper right'
                           )
        else:
            self.hlegend = self.ax.legend( (self.hrho, self.hrhoI),
                                           (r'$\rho$', r'$\rho_i$'),
                                           loc = (0.85,0.5)
                                           #loc='upper right'
                                           )
        self.hlegend.get_frame().set( alpha=0.2, facecolor='yellow' )
        self.axes_frozen = False


        # update the figure
        self.update()



    def ShowRho(self, show=True):
        self.hrho.set_visible(show)

    def ShowRhoI(self, show=True):
        self.hrhoI.set_visible(show)

    def ShowRhoM(self, show=True):
        self.hrhoM.set_visible(show)

    def ShowThetaM(self, show=True):
        self.hthetaM.set_visible(show)

    def ShowLegend(self, show):
        self.hlegend.set_visible(show)

    def onMotion(self, event):
        """Respond to motion events by changing the active layer."""
        # Find the layer containing the mouse
        if self.layer:
            # Quick rejection if it is in the layer
            if self.layer_start < event.xdata < self.layer_end:
                #print "quick reject", event.xdata, self.layer_start, self.layer_end
                return False
            # If it is within 5 pixels of the layer, keep to the same layer
            transform = self.ax.transData
            xlo,_ = pixel_to_data(transform, event.x+pick_radius, event.y)
            xhi,_ = pixel_to_data(transform, event.x-pick_radius, event.y)
            if self.layer_start < xlo and xhi < self.layer_end:
                #print "hysteresis", event.x, xlo, xhi, self.layer_start, self.layer_end
                return False
        
        # TODO: may want some hi
        self.set_layer(event.xdata)
        
        return False


    def onContext(self, ev):
        """Context menu (eventually ...)."""
        return False

    def set_layer(self, z):
        """Make layer containing z the active layer."""
        # Check if the markers are already set
        layer,start,end = self.experiment.sample.find(z)
        #print "moving to layer",layer,start,end
        self.layer = layer
        if end == 0:
            start, end = -inf, 0
        elif start == self.experiment.sample.thickness.value:
            start, end = start, inf
        self.layer = layer
        self.layer_start, self.layer_end = start, end
 
        # Clear the old markers
        if self.segment_interactor:
            self.segment_interactor.clear_markers()
        #self.segment_interactor = make_interactor(self, layer)

        self.draw()


    def update(self):
        """
        Respond to changes in the model by recalculating the profiles and
        resetting the widgets.
        """
        # We are done the manipulation; let the model send its update signal
        # to whomever is listening.
        self.listener.signal('update',self)

        # Update locations
        self.thickness_interactor.update()
        self.interface_interactor.update()
        if self.segment_interactor:
            self.segment_interactor.update()

        # Update profile
        self.experiment.update()
        if self.magnetic:
            z,rho,rhoI,rhoM,thetaM = self.experiment.smooth_profile()
            self.hrhoM.set_data(z,rhoM)
            self.hthetaM.set_data(z,thetaM)
        else:
            z,rho,rhoI = self.experiment.smooth_profile()
        self.hrho.set_data(z,rho)
        self.hrhoI.set_data(z,rhoI)


        # Compute automatic y limits
        # Note: theta limits are on ax2

        # TODO: the markers themselves may fall outside the graph.
        # May need to walk sample to find the position of the markers
        # when setting the limits.

        if self.magnetic:
            lo = min( rho.min(), rhoI.min(), rhoM.min() )
            hi = max( rho.max(), rhoI.max(), rhoM.max() )
            fluff = 0.1*(hi-lo)
            self.ylim = lo-fluff, hi+fluff
            lo, hi = thetaM.min(), thetaM.max()
            fluff = 0.1*(hi-lo)
            self.y2lim = lo-fluff, hi+fluff
        else:
            lo = min( rho.min(), rhoI.min() )
            hi = max( rho.max(), rhoI.max() )
            fluff     = 0.05*(hi-lo)
            self.ylim = lo-fluff, hi+fluff
        self.xlim = z[0],z[-1]

        self.draw()

    def freeze_axes(self):
        self.axes_frozen = True

    def thaw_axes(self):
        self.axes_frozen = False

    def draw(self):
        """Set the limits and tell the canvas to render itself."""
        # TODO: Stop doing surprising things with limits
        # TODO: Detect if user is zoomed, and freeze limits if that is the case

        if  not self.axes_frozen:
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)
            if self.magnetic:
                self.ax2.set_ylim(self.y2lim)

        self.ax.figure.canvas.draw_idle()
