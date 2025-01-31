"""
Reflectometry profile interactor.
"""

import math

import numpy as np
from matplotlib import transforms
from numpy import inf

from . import registry
from .binder import BindArtist, pixel_to_data
from .config import layer_hysteresis, rho_color, rhoI_color, rhoM_color, thetaM_color
from .interactor import BaseInteractor, safecall
from .interface import InterfaceInteractor
from .thickness import ThicknessInteractor

blend_xy = transforms.blended_transform_factory


# ================== Main profile interactor =====================
class ProfileInteractor(object):
    """
    Reflectometry profile editor
    """

    _debug = False

    def __init__(self, axes, theta_axes, status_update):
        self.status_update = status_update
        self.canvas = axes.figure.canvas
        self.axes = axes
        self.theta_axes = theta_axes
        self.xcoords = blend_xy(axes.transData, axes.transAxes)

        # Add some plots
        [self.hrho] = axes.plot([], [], "-", color=rho_color, label=r"$\rho$")
        [self.hrhoI] = axes.plot([], [], "-", color=rhoI_color, label=r"$\rho_i$")
        [self.hrhoM] = self.axes.plot([], [], "-", color=rhoM_color, label=r"$\rho_M$")
        [self.hthetaM] = self.theta_axes.plot([], [], "-", color=thetaM_color, label=r"$\theta_M$")

        # Ick! trying to do motion event before profile has been set
        self.experiment = None

        # TODO: the connect mechanism needs to be owned by the canvas rather
        # than the axes --- cannot have multiple profiles on the same canvas
        # until connect is in the right place.
        self.connect = BindArtist(axes.figure)
        self.connect.clearall()
        self.connect("motion", axes, self.onMotion)
        self.connect("click", axes.figure, self.onClick)

        # Set interactors
        self.thickness_interactor = ThicknessInteractor(self)
        self.interface_interactor = InterfaceInteractor(self)
        self.layer_interactor = BaseInteractor(self)

    def set_experiment(self, experiment, force_recalc, signal_update):
        self.experiment = experiment
        self.force_recalc = force_recalc
        self.signal_update = signal_update

        # Theta needs a separate axis, we put these two axes into a figure
        self.axes.set_xlabel(r"$\rm{z}/\AA$")
        self.ismagnetic = experiment.sample.ismagnetic
        self.isstep = experiment.step_interfaces
        if self.ismagnetic:
            self.axes.set_ylabel(r"$\rm{SLD}\ \rho,\  \rho_i,\  \rho_M\ /\ 10^{-6} \AA^{-2}$")
            self.theta_axes.set_ylabel(r"$\rm{Magnetic\ Angle}\ \theta_M\ /\ ^\circ$")
            self.theta_axes.set_visible(True)
            self.hrhoM.set_visible(True)
            self.hthetaM.set_visible(True)
            self.hlegend = self.axes.legend(
                (self.hrho, self.hrhoI, self.hrhoM, self.hthetaM),
                # ('SLD','Absorption','Mag. SLD','Mag. angle'),
                (r"$\rho$", r"$\rho_i$", r"$\rho_M$", r"$\theta_M$"),
                loc=(0.85, 0.5),
                # loc='upper right'
            )
        else:
            self.axes.set_ylabel(r"$\rm{SLD}\ \rho,\  \rho_i\ /\ 10^{-6} \AA^{-2}$")
            self.theta_axes.set_ylabel("")
            self.theta_axes.set_visible(False)
            self.hrhoM.set_visible(False)
            self.hthetaM.set_visible(False)
            self.hlegend = self.axes.legend(
                (self.hrho, self.hrhoI),
                (r"$\rho$", r"$\rho_i$"),
                loc=(0.85, 0.5),
                # loc='upper right'
            )
        self.hlegend.get_frame().set(alpha=0.2, facecolor="yellow")

        self.x_offset = 0
        self.layer_num = None
        self._find_layer_boundaries()
        self.thickness_interactor.reset_markers()
        self.set_layer(0)
        self.redraw()

    def update_cursor(self, x, y):
        def nice(value, range):
            place = int(math.log10(abs(range[1] - range[0])) - 3)
            # print value,range,place
            if place < 0:
                return "%.*f" % (-place, value)
            else:
                return "%d" % int(value)

        self.status_update("x:%s  y:%s" % (nice(x, self.axes.get_xlim()), nice(y, self.axes.get_ylim())))

    def onMotion(self, event):
        """Respond to motion events by changing the active layer."""
        if self.experiment is None:
            return False

        # Force data coordinates for the mouse position
        transform = self.axes.transData
        x, y = pixel_to_data(transform, event.x, event.y)
        self.update_cursor(x, y)

        # Check if mouse is still in the same layer
        if self.layer_num is not None:
            left, right = self.boundary[self.layer_num : self.layer_num + 2]
            # Quick rejection if it is in the layer
            if left < x < right:
                # print "quick reject", event.xdata, self.layer_start, self.layer_end
                return False
            # If it is within 5 pixels of the layer, keep to the same layer
            xlo, _ = pixel_to_data(transform, event.x + layer_hysteresis, event.y)
            xhi, _ = pixel_to_data(transform, event.x - layer_hysteresis, event.y)
            if left < xlo and xhi < right:
                # print "hysteresis", event.x, xlo, xhi, self.layer_start, self.layer_end
                return False

        # If layer changed, then replace the markers
        self.set_layer(x)
        return False

    def onClick(self, ev):
        # Couldn't make popups work --- moved context menu to panel
        return False

    # === Sample information ===
    def find(self, z):
        return self.experiment.sample.find(z - self.x_offset)

    def _find_layer_boundaries(self):
        offset = self.x_offset
        boundary = [-inf, offset]
        if hasattr(self.experiment, "sample"):
            for L in self.experiment.sample[1:-1]:
                dx = L.thickness.value
                offset += dx
                boundary.append(offset)
        boundary.append(inf)
        self.boundary = np.asarray(boundary)

    def sample_labels(self):
        return [L.name for L in self.experiment.sample]

    def sample_layer(self, n):
        return self.experiment.sample[n]

    @safecall
    def set_layer(self, z):
        """Make layer containing z the active layer."""
        # Note: sample_layer needs layer_num to be an int, rather than a
        # numpy int64, since it tests for int vs. string vs. material.
        self.layer_num = int(np.searchsorted(self.boundary, z) - 1)
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
        if self.ismagnetic:
            z, rho, rhoI, rhoM, thetaM = self.experiment.magnetic_smooth_profile()
            self.hrhoM.set_data(z, rhoM)
            self.hthetaM.set_data(z, thetaM)
        else:
            z, rho, rhoI = self.experiment.smooth_profile()
        # print "z",z
        # print "offset",self.x_offset
        self.hrho.set_data(z + self.x_offset, rho)
        self.hrhoI.set_data(z + self.x_offset, rhoI)

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
        self.axes.set_xlim(z[0], z[-1])

        if self.ismagnetic:
            rhoM = self.hrhoM.get_ydata()
            thetaM = self.hthetaM.get_ydata()
            lo = min(rho.min(), rhoI.min(), rhoM.min())
            hi = max(rho.max(), rhoI.max(), rhoM.max())
            fluff = 0.05 * (hi - lo)
            self.axes.set_ylim(lo - fluff, hi + fluff)

            lo, hi = thetaM.min(), thetaM.max()
            fluff = 0.05 * (hi - lo)
            if fluff <= 1e-3:
                lo, hi, fluff = 0, 360, 0
            self.theta_axes.set_ylim(lo - fluff, hi + fluff)

        else:
            lo = min(rho.min(), rhoI.min())
            hi = max(rho.max(), rhoI.max())
            fluff = 0.05 * (hi - lo)
            self.axes.set_ylim(lo - fluff, hi + fluff)

    def update(self):
        """
        Respond to changes in the model by recalculating the profiles and
        resetting the widgets.
        """
        # We are done the manipulation; let the model send its update signal
        # to whomever is listening.
        self.force_recalc()
        self.redraw()
        self.signal_update()

    def redraw(self, reset_limits=False):
        self.update_markers()
        self.update_profile()
        if reset_limits:
            self.reset_limits()
        self.draw_now()

    def draw_now(self):
        # print "draw immediately"
        self.canvas.draw()

    def draw_idle(self):
        """Set the limits and tell the canvas to render itself."""
        # print "draw when idle"
        self.canvas.draw_idle()
