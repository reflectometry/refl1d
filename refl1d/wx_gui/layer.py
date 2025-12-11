"""
Layer interactor.
"""

from bumps.parameter import Parameter

from refl1d.sample.material import SLD
from .interactor import BaseInteractor
from .util import setpar
from .config import pick_radius
from .config import rho_color, rhoI_color, disabled_color


class MaterialInteractor(BaseInteractor):
    def __init__(self, profile, material, range=(0, 1)):  # @ReservedAssignment
        super(MaterialInteractor, self).__init__(profile)
        self.material = material
        self.range = range

        if isinstance(material, SLD):
            enabled = True
            self._rho = material.rho
            self._rhoI = material.irho
            self._rho_scale = 1
            self._rhoI_scale = 1
        elif hasattr(material, "density") and isinstance(material.density, Parameter):
            enabled = True
            rho, rhoI = material.sld(profile.experiment.probe)
            self._rho = material.density
            self._rhoI = material.density
            self._rho_scale = 1 / (rho / material.density.value)
            self._rhoI_scale = 1 / (rhoI / material.density.value)
        else:
            self._rho = self._rhoI = None
            enabled = False

        style = dict(
            linestyle="--",
            linewidth=2,
            pickradius=pick_radius,
        )
        ax = profile.axes
        rho = ax.plot(
            [], [], label=material.name + " rho", color=rho_color if enabled else disabled_color, zorder=6, **style
        )[0]
        rhoI = ax.plot(
            [], [], label=material.name + " rhoI", color=rhoI_color if enabled else disabled_color, zorder=5, **style
        )[0]

        self.markers = [rho, rhoI]
        if enabled:
            self.connect_markers(self.markers)

    def update_markers(self):
        z = self.profile.boundary[1:-1]
        n = self.profile.layer_num
        # if n is None: return
        if n == 0:
            left, right = -20, 0
        elif n >= len(z):
            left, right = z[-1], z[-1] + 20
        else:
            delta = z[n] - z[n - 1]
            left, right = self.range[0] * delta + z[n - 1], self.range[1] * delta + z[n - 1]

        rho, rhoI = self.material.sld(self.profile.experiment.probe)
        self.markers[0].set_data((left, right), (rho, rho))
        self.markers[1].set_data((left, right), (rhoI, rhoI))

    def save(self, ev):
        if self._rho is not None:
            self._rho_save = self._rho.value
        if self._rhoI is not None:
            self._rhoI_save = self._rhoI.value

    def restore(self, ev):
        if self._rho is not None:
            self._rho.value = self._rho_save
        if self._rhoI is not None:
            self._rhoI.value = self._rhoI_save

    def drag(self, ev):
        if ev.artist == self.markers[0]:
            setpar(self._rho, ev.ydata * self._rho_scale)
        else:
            setpar(self._rhoI, ev.ydata * self._rhoI_scale)


class SlabInteractor(BaseInteractor):
    def __init__(self, profile, layer):
        super(SlabInteractor, self).__init__(profile)
        self.material = MaterialInteractor(profile, layer.material)

    def update_markers(self):
        self.material.update_markers()

    def clear_markers(self):
        self.material.clear_markers()


# -------------------------------------------------------------------
class NoInteractor(BaseInteractor):
    """
    Null Interactor for undefined layers.
    """

    def __init__(self, profile, layer):
        super(NoInteractor, self).__init__(profile)

    # TODO: turn layer.parameters() into sliders
