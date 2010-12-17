"""
Basic interactor for Reflectometry profile.
"""

from matplotlib      import transforms

from binder       import pixel_to_data
from config       import active_color

try:    
    # matplotlib 0.98
    blend_xy = transforms.blended_transform_factory
except: 
    # CRUFT: matplotlib 0.91
    blend_xy = transforms.blend_xy_sep_transform

# ---------------------- Base interactors ----------------------
# GUI starts here
# Other interactor will inherit this
# --------------------------------------------------------------

class BaseInteractor(object):
    """
    Common base class for profile interactors

    Individual interactors need the following functions:

        save(ev)  - save the current state for later restore
        restore() - restore the old state
        move(x,y,ev) - move the interactor to position x,y
        moveend(ev) - end the drag event
        update() - draw the interactors

    The following are provided by the base class:

        connect_markers(markers) - register callbacks for all markers
        clear_markers() - remove all items in self.markers
        onHilite(ev) - enter/leave event processing
        onLeave(ev) - enter/leave event processing
        onClick(ev) - mouse click: calls save()
        onRelease(ev) - mouse click ends: calls moveend()
        onDrag(ev) - mouse move: calls move() or restore()
        onKey(ev) - keyboard move: calls move() or restore()

    Interactor attributes:

        profile  - the profile we are a part of
        axes  - axes holding the interactor
        color - color of the interactor in non-active state
        markers - list of handles for the interactor
    """
    def __init__(self,
                 profile,
                 axes,
                 color='black'
                 ):
        self.profile  = profile
        self.axes  = axes
        self.xcoords  = blend_xy(axes.transData, axes.transAxes)
        self.color         = color
        self._save_n       = 0
        self._save_depth_n = 0
        self.click_flag    = 0

    def connect_markers(self,markers):
        """
        Connect markers to callbacks
        """
        for h in markers:
            connect = self.profile.connect
            connect('enter',   h, self.onHilite)
            connect('leave',   h, self.onLeave)
            connect('click',   h, self.onClick)
            connect('release', h, self.onRelease)
            connect('drag',    h, self.onDrag)
            connect('key',     h, self.onKey)

    def clear_markers(self):
        """Clear old markers and interfaces."""
        for h in self.markers:
            h.remove()
        if self.markers:
            self.profile.connect.clear(*self.markers)
        self.markers = []


    #======================================
    def save(self, ev):
        pass

    def restore(self, ev):
        pass

    def move(self, x, y, ev):
        pass

    def moveend(self, ev):
        pass

    def updateValue(self, event):
        pass

    def setValue(self, event):
        pass

    #=====================================



    def onHilite(self, event):
        """
        Hilite the artist reporting the event, indicating that it is
        ready to receive a click.
        """
        event.artist.set_color(active_color)
        self.profile.draw()
        self.click_flag = 0

        return True


    def onLeave(self, event):
        """
        Restore the artist to the original colour when the cursor leaves.
        """
        event.artist.set_color(self.color)
        self.profile.draw()
        return True


    def onClick(self, event):
        """
        Prepare to move the artist.  Calls save() to preserve the state for
        later restore().
        """
        # set the click_flag
        self.click_flag  = 1


        # Make sure the x,y data use the coordinate system of the
        # artist rather than the default axes coordinates.
        transform = event.artist.get_transform()
        xy = pixel_to_data(transform, event.x, event.y)
        event.xdata, event.ydata = xy

        # save current position and event
        self.clickx = event.xdata
        self.clicky = event.ydata
        self.save(event)

        # save the layer number for current mouse position
        self._save_n = self.profile.find( event.xdata )

        # save best layer index for current mouse position
        self._save_depth_n = self.BestDepthLayerNum(event.xdata)

        # Here we just show the parameter. Change nothing to model
        self.showValue(event)


        return True



    def onRelease(self, event):
        """
        Release the mouse
        """
        self.moveend(event)

        # Release the click flag for next click operation
        self.click_flag = 0

        return True


    def onDrag(self, event):
        """
        Move the artist.  Calls move() to update the state, or restore() if
        the mouse leaves the window.
        """

        inside,_ = self.axes.contains(event)

        if  inside:
            self.clickx = event.xdata
            self.clicky = event.ydata

            # In case of no click and just drag
            if self.click_flag == 0:
                self._save_depth_n =  self.BestDepthLayerNum(event.xdata)

            # Why?
            self.click_flag = 1

            self.move(event.xdata, event.ydata, event)

            # set the parameter
            self.setValue(event)

        else:
            self.restore()

        #update model
        self.profile.update()

        return True



    def onKey(self, event):
        """
        Respond to keyboard events.  Arrow keys move the widget.  Escape
        restores it to the position before the last click.

        Calls move() to update the state.  Calls restore() on escape.
        """
        if event.key == 'escape':
            self.restore()

        elif event.key in ['up', 'down', 'right', 'left']:

            #inside,prop = self.axes.contains(event)
            if  hasattr(self, 'clickx') and hasattr(self, 'clicky'):

                dx,dy=self.dxy(self.clickx,self.clicky,
                               step=0.2 if event.control else 1)

                if   event.key == 'up':    self.clicky += dy
                elif event.key == 'down':  self.clicky -= dy
                elif event.key == 'right': self.clickx += dx
                elif event.key == 'left':  self.clickx -= dx


                # update the state
                self.move(self.clickx, self.clicky, event)

        else:
            return False

        return True



    def dxy(self,x,y,step=1):
        """
        Return the step size in data coordinates for a small
        step in screen coordinates.
        """
        ax = self.axes
        px,py = ax.transData.inverted().transform( (x,y) )
        nx,ny = ax.transData.xy_tup( (px+step, py+step) )
        dx = nx-x
        dy = ny-y

        return dx,dy
