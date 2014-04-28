"""
Basic interactor for Reflectometry profile.
"""

from .binder import pixel_to_data
from .config import active_color


def safecall(fn):
    def wrapped(self, *args, **kw):
        try:
            fn(self, *args, **kw)
        except:
            if self._debug:
                import sys, traceback
                traceback.print_exc()
                sys.exit(1)
            else:
                raise
    return wrapped

# ---------------------- Base interactors ----------------------
# GUI starts here
# Other interactor will inherit this
# --------------------------------------------------------------

class BaseInteractor(object):
    """
    Common base class for profile interactors

    Individual interactors need the following functions:

        save(ev)    - save the current state for later restore
        restore(ev) - restore the old state
        drag(ev)    - move the interactor to position ev.xdata,ev.ydata
        update_markers() - redraw the interactors

    If something special is required while dragging:

        drag_start(ev)  - drag started
        drag_cancel(ev) - drag cancelled with escape key
        drag_done(ev)   - drag completed

    The following are provided by the base class:

        connect_markers(markers) - register callbacks for all markers
        clear_markers() - remove all items in self.markers

    The actions associated with the interactor are:
        _onHilite(ev) - enter/leave event processing
        _onLeave(ev) - enter/leave event processing
        _onClick(ev) - mouse click: calls save()
        _onRelease(ev) - mouse click ends: calls moveend()
        _onDrag(ev) - mouse move: calls move() or restore()
        _onKey(ev) - keyboard move: calls move() or restore()

    Interactor attributes:

        profile  - the profile containing the interaactor
        color    - color of the interactor in non-active state
        markers  - list of handles for the interactor
    """
    _debug = False
    def __init__(self, profile):
        self.profile = profile
        self.markers = []

        # Remember if we are dragging during motion
        self._dragging = False

        # Remember the last location for keyboard dragging
        self._arrow_x, self._arrow_y = None, None

    def connect_markers(self, markers):
        """
        Connect markers to callbacks
        """
        connect = self.profile.connect
        for h in markers:
            connect('enter',   h, self._onEnter)
            connect('leave',   h, self._onLeave)
            connect('click',   h, self._onClick)
            connect('release', h, self._onRelease)
            connect('drag',    h, self._onDrag)
            connect('key',     h, self._onKey)

    def clear_markers(self):
        """Clear old markers and interfaces."""
        connect = self.profile.connect
        for h in self.markers:
            h.remove()
        if self.markers:
            connect.clear(*self.markers)
        self.markers = []


    #======================================
    def update_markers(self):
        """
        Configuration has changed: update marker positions
        """
    def save(self, ev):
        """
        Save the current state.

        Save is triggered by both a mouse click and a mouse release.
        """
    def restore(self, ev):
        """
        Restore the saved state
        """
    def drag(self, ev):
        """
        Move the interactor relative to the current state.

        Drag operations are usually triggered by the mouse, using the following
        call pattern::

            save()
            drag_start()
            drag()
            drag()
            ...
            drag_done()
            save()

        The final save allows keyboard actions to happen on the current
        interactor after the mouse manipulation has stopped.  Note that
        the drag actions are happening outside the drag_start/drag_done.
        Note also that the event itself will be marked as a keyboard
        event.

            save()
            drag_start() # mouse click
            drag_done()  # mouse release
            save()
            drag()       # arrow key
            drag()       # arrow key
            drag()       # arrow key
            restore()    # escape key

        When the escape key is pressed during drag, or the mouse leaves the
        screen, the drag may be cancelled::

            save()
            drag_start()
            drag()
            drag()
            ...
            drag_cancel()
            restore()
        """
    def drag_start(self, ev):
        """
        Start drag.

        Normally don't need to do anything here since save is triggered
        automatically.
        """
    def drag_cancel(self, ev):
        """
        Cancel drag.

        Normally don't need to do anything since restore is triggered
        automatically.
        """

    def drag_done(self, ev):
        """
        End drag.

        Normally don't need to do anything since the value was updated
        by drag.
        """

    #======================================#
    # Event processors --- do not override #
    #======================================#
    def _onEnter(self, event):
        """
        Hilite the artist reporting the event, indicating that it is
        ready to receive a click.
        """
        self._color = event.artist.get_color()
        event.artist.set_color(active_color)
        # TODO: use overlay on Hilite and remove it on Leave
        self.profile.draw_idle()

        return True


    def _onLeave(self, event):
        """
        Restore the artist to the original colour when the cursor leaves.
        """
        event.artist.set_color(self._color)
        self.profile.draw_idle()
        return True


    @safecall
    def _onClick(self, event):
        """
        Prepare to move the artist.  Calls save() to preserve the state for
        later restore().
        """
        # Make sure the x,y data use the coordinate system of the
        # artist rather than the default axes coordinates.
        transform = event.artist.get_transform()
        xy = pixel_to_data(transform, event.x, event.y)
        event.xdata, event.ydata = xy

        # save current position and event
        self.save(event)

        self.drag_start(event)
        self._dragging = True

        return True

    @safecall
    def _onRelease(self, event):
        """
        Release the mouse
        """
        # We are done the click-drag operation
        self.drag_done(event)
        self._dragging = False
        #self.profile.draw_idle()

        # Prepare for keyboard adjustment
        self._arrow_trans = event.artist.get_transform()
        self._arrow_x = event.xdata
        self._arrow_y = event.ydata

        return True


    @safecall
    def _onDrag(self, event):
        """
        Move the artist.  Calls move() to update the state, or restore() if
        the mouse leaves the window.
        """

        inside,_ = self.profile.axes.contains(event)

        if  inside:
            # In case of no click and just drag
            if not self._dragging:
                print("We have a drag without a click")
                self._dragging = True
            self.drag(event)
        else:
            # Dragging left the canvas: undo the change
            self.drag_cancel(event)
            self.restore(event)

        # update model
        self.profile.update()

        return True

    @safecall
    def _onKey(self, event):
        """
        Respond to keyboard events.  Arrow keys move the widget.  Escape
        restores it to the position before the last click.

        Calls move() to update the state.  Calls restore() on escape.

        Returns True if key is handled
        """
        if event.key == 'escape':
            if self._dragging:
                self.drag_cancel(event)
            self.restore(event)
            self.profile.update()
            return True

        elif event.key in ['up', 'down', 'right', 'left']:

            if self._dragging:
                return True

            if  not (hasattr(self, '_arrow_x') and hasattr(self, '_arrow_y')):
                print("got key event without having location")
                return True

            step = 0.2 if event.control else 1
            x,y = self._arrow_x, self._arrow_y
            px,py = self._arrow_trans.transform_point( (x,y) )
            if   event.key == 'up':    py += step
            elif event.key == 'down':  py -= step
            elif event.key == 'right': px += step
            elif event.key == 'left':  px -= step
            x,y = self._arrow_trans.inverted().transform_point( (px,py) )
            self._arrow_x, self._arrow_y = x,y


            # update the state
            event.xdata, event.ydata = x,y
            # TODO: sound bell when at limits
            self.drag(event)
            self.profile.update()

            return True

        else:
            return False

        raise RuntimeError("Unreachable code")
