import numpy as np

from .config import pick_radius
from .config import profile_color
from .config import vf_scale
from .interactor import BaseInteractor
from .layer import MaterialInteractor
from .util import clip, setpar


# -----------------------------------------------------------------------
class FreeInterfaceInteractor(BaseInteractor):
    """
    Interactor for freeform interfaces using monotonic splines.
    """

    def __init__(self, profile, layer):
        super(FreeInterfaceInteractor, self).__init__(profile)
        self.below = MaterialInteractor(profile, layer.below, range=(0, 0.3))
        self.above = MaterialInteractor(profile, layer.above, range=(0.7, 1))
        self.layer = layer
        ax = profile.axes

        style = dict(
            linestyle=":",
            transform=profile.xcoords,
            zorder=0,
            color=profile_color,
            visible=True,
        )
        self.hprofile = ax.plot([], [], **style)[0]

        style = dict(
            marker="o",
            transform=profile.xcoords,
            zorder=7,
            pickradius=pick_radius,
            color=profile_color,
            alpha=0.5,
            markersize=5,
            visible=True,
        )
        self.markers = [ax.plot([], [], **style)[0] for _ in layer.dz[:-1]]
        self.connect_markers(self.markers)

    def update_markers(self):
        """
        Draw the widgets in their new positions.
        """
        self.below.update_markers()
        self.above.update_markers()

        n = self.profile.layer_num
        left = self.profile.boundary[n]
        layer = self.layer

        z = np.linspace(0, layer.thickness.value, 200)
        vf = layer.profile(z)
        self.hprofile.set_xdata(z + left)
        self.hprofile.set_ydata(vf * vf_scale)

        z = np.cumsum(np.array([v.value for v in layer.dz], "d"))
        p = np.cumsum(np.array([v.value for v in layer.dp], "d"))
        self._zscale = layer.thickness.value / z[-1]
        if p[-1] == 0:
            p[-1] = 1
        p *= vf_scale / p[-1]
        z *= self._zscale
        z += left
        for h, zi, pi in zip(self.markers, z[:-1], p[:-1]):
            h.set_data((zi,), (pi,))
        # self.markers[0].set_data(z,p)

    def clear_markers(self):
        self.below.clear_markers()
        self.above.clear_markers()
        super(FreeInterfaceInteractor, self).clear_markers()
        self.hprofile.remove()

    def drag(self, evt):
        """
        Update the model with the new widget position.
        """
        n = self.profile.layer_num
        left, right = self.profile.boundary[n : n + 2]

        idx = self.markers.index(evt.artist)
        z = [h.get_xdata()[0] for h in self.markers]
        z[idx] = evt.xdata
        dz = np.diff(np.sort(np.clip(z + [left, right], left, right)))
        dz /= np.max(dz)
        for p, v in zip(self.layer.dz, dz):
            setpar(p, v)

    def save(self, evt):
        """
        Save the current state of the model represented by the widget.
        """
        self._save_dz = [p.value for p in self.layer.dz]
        self._save_dp = [p.value for p in self.layer.dp]

    def restore(self, evt):
        """
        Restore the widget and model to the saved state.
        """
        for p, v in zip(self.layer.dz, self._save_dz):
            p.value = v
        for p, v in zip(self.layer.dp, self._save_dp):
            p.value = v


# -----------------------------------------------------------------------
class FreeLayerInteractor(BaseInteractor):
    """
    Interactor for freeform layers using monotonic splines.
    """

    def __init__(self, profile, layer):
        super(FreeLayerInteractor, self).__init__(profile)
        self.below = MaterialInteractor(profile, layer.below, range=(0, 0.3))
        self.above = MaterialInteractor(profile, layer.above, range=(0.7, 1))
        self.layer = layer
        ax = profile.axes

        style = dict(
            marker="o",
            zorder=7,
            pickradius=pick_radius,
            color=profile_color,
            alpha=0.5,
            markersize=5,
            visible=True,
        )
        self.markers = [ax.plot([], [], **style)[0] for _ in layer.rho + layer.irho]
        self.connect_markers(self.markers)

    def update_markers(self):
        """
        Draw the widgets in their new positions.
        """
        self.below.update_markers()
        self.above.update_markers()

        n = self.profile.layer_num
        left = self.profile.boundary[n]
        layer = self.layer

        z = np.array([v.value for v in layer.z])
        rho = np.array([v.value for v in layer.rho])
        rhoI = np.array([v.value for v in layer.irho])
        self._zscale = layer.thickness.value
        z *= self._zscale
        z += left
        for h, zi, pi in zip(self.markers[: len(rho)], z, rho):
            h.set_data((zi,), (pi,))
        for h, zi, pi in zip(self.markers[len(rho) :], z, rhoI):
            h.set_data((zi,), (pi,))

    def clear_markers(self):
        self.below.clear_markers()
        self.above.clear_markers()
        super(FreeLayerInteractor, self).clear_markers()

    def drag(self, evt):
        """
        Update the model with the new widget position.
        """
        n = self.profile.layer_num
        left, right = self.profile.boundary[n : n + 2]
        z, rho, rhoI = self.layer.z, self.layer.rho, self.layer.irho
        nrho = len(rho)

        zval = clip((evt.xdata - left) / (right - left), 0, 1)
        idx = self.markers.index(evt.artist)
        if idx >= nrho:
            idx -= nrho
            setpar(z[idx], zval)
            setpar(rhoI[idx], evt.ydata)
        else:
            setpar(z[idx], zval)
            setpar(rho[idx], evt.ydata)

    def save(self, evt):
        """
        Save the current state of the model represented by the widget.
        """
        self._save_z = [p.value for p in self.layer.z]
        self._save_rho = [p.value for p in self.layer.rho]
        self._save_rhoI = [p.value for p in self.layer.irho]

    def restore(self, evt):
        """
        Restore the widget and model to the saved state.
        """
        for p, v in zip(self.layer.z, self._save_z):
            p.value = v
        for p, v in zip(self.layer.rho, self._save_rho):
            p.value = v
        for p, v in zip(self.layer.irho, self._save_rhoI):
            p.value = v
