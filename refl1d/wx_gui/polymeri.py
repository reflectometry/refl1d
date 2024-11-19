from math import log

import numpy

from .interactor import BaseInteractor
from .layer import MaterialInteractor
from .util import setpar


# -----------------------------------------------------------------------
class PolymerBrushInteractor(BaseInteractor):
    """
    Interactor for tethered polymer model.
    """

    def __init__(self, profile, layer):
        super(PolymerBrushInteractor, self).__init__(profile)
        self.polymer = MaterialInteractor(profile, layer.polymer, range=(0, 0.3))
        self.solvent = MaterialInteractor(profile, layer.solvent, range=(0.7, 1))
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
            linestyle="-",
            transform=profile.xcoords,
            zorder=4,
            pickradius=pick_radius,
            color=profile_color,
            linewidth=1,
            visible=True,
        )
        self.hphi = ax.plot([], [], **style)[0]
        self.hbase = ax.plot([], [], **style)[0]
        self.hlength = ax.plot([], [], **style)[0]

        style = dict(
            marker="s",
            transform=profile.xcoords,
            zorder=5,
            pickradius=pick_radius,
            color=profile_color,
            alpha=0.5,
            markersize=5,
            visible=True,
        )
        self.hpower = ax.plot([], [], **style)[0]
        self.hsigma = ax.plot([], [], **style)[0]
        # print "xcoords",id(profile.xcoords),"trans",id(self.hbase.get_transform())

        self.markers = [self.hphi, self.hbase, self.hlength, self.hpower, self.hsigma]
        self.parameters = [layer.base_vf, layer.base, layer.length, layer.power, layer.sigma]
        self.connect_markers(self.markers)

    def clear_markers(self):
        super(PolymerBrushInteractor, self).clear_markers()
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
        left = self.profile.boundary[n]
        layer = self.layer

        # TODO: want to use smae step size for profile as we do for
        # smooth profiles  Don't know how to access that value nicely.
        z = numpy.arange(0, layer.thickness.value, 0.1)
        vf = layer.profile(z)
        self.hprofile.set_data(z + left, vf * vf_scale)

        # at z midway between l1 and l2
        #     y = phi * 0.75**p
        phi = layer.base_vf.value * vf_scale / 100
        phi_power = phi * 0.75**layer.power.value
        L0 = left + layer.base.value
        L1 = L0 + layer.length.value
        Lpower = (L0 + L1) / 2
        Lsigma = L1 + layer.sigma.value

        # print "L0,L1,Lpower,Lsigma",L0,L1,Lpower,Lsigma,phi,phi_power
        self.hphi.set_data((left, L0), (phi, phi))
        self.hbase.set_data((L0, L0), (0, phi))
        self.hlength.set_data((L1, L1), (0, phi))
        self.hpower.set_data((Lpower,), (phi_power,))
        self.hsigma.set_data((Lsigma,), (0,))

    def drag(self, ev):
        """
        Update the model with the new widget position.
        """
        table = dict(zip(self.markers, self.parameters))
        par = table[ev.artist]
        n = self.profile.layer_num
        left = self.profile.boundary[n]
        right = self.profile.boundary[n + 1]
        # TODO: Keeping individual interactors within the layer does not
        # necessarily keep the whole interactor in the layer.
        z = max(left, min(right, ev.xdata))
        if ev.artist == self.hphi:
            # print ev.ydata, self.profile.xcoords.inverted().transform([(ev.x,ev.y)])[0][1]
            setpar(par, ev.ydata * 100 / vf_scale)
            # print "phi",par.value,ev.ydata*100/vf_scale
        elif ev.artist == self.hbase:
            offset = left
            setpar(par, z - offset)
        elif ev.artist == self.hlength:
            offset = left + self.layer.base.value
            setpar(par, z - offset)
        elif ev.artist == self.hsigma:
            offset = left + self.layer.base.value + self.layer.length.value
            setpar(par, z - offset)
        elif ev.artist == self.hpower:
            phi = self.layer.base_vf.value * vf_scale / 100
            phi_power = log(ev.ydata / phi) / log(0.75) if ev.ydata > 0 else 100
            setpar(par, phi_power)

    def save(self, ev):
        """
        Save the current state of the model represented by the widget.
        """
        table = dict(zip(self.markers, self.parameters))
        par = table[ev.artist]
        self._save_par = par
        self._save_val = par.value

    def restore(self, ev):
        """
        Restore the widget and model to the saved state.
        """
        self._save_par.value = self._save_val
