"""
Simulated error-in-variables fit for reflectometry.

Data is simulated for a reflectivity curve with jitter in the motor angle
relative to the nominal angle reported by the motor encoders.

Adjust the parameters and uncertainties for the simulation in the model setup
below. Choose *fit_strategy = 'nominal'* for a traditional fit, or *'eiv'* for
an error-in-variables fit.
"""

import numpy as np

from refl1d.names import *

# ********** model setup **********

# Define the material model
nickel = Material("Ni")
sample = silicon(0, 5) | nickel(100, 5) | air

# Set the fitting parameters
sample[0].interface.range(0, 20)
sample[1].interface.range(0, 20)
sample[1].thickness.range(0, 400)

# Set the probe parameters, determining number of points and resolution
# n = 20
n = 400
dT, L, dL = 0.02, 4.75, 0.0475
T = np.linspace(0, 5, n)

# Set the motor uncertainty and the measurement uncertainty.
angle_uncertainty = 0.005
angle_distribution = "uniform"
# angle_distribution = 'gaussian'
refl_error = 0.1  # measurement uncertainty in [0, 100]

# fit_strategy = 'nominal'
fit_strategy = "eiv"

# ************* done model setup **************


# ===== code for error-in-variables fitting =====
def marginalized_residuals(Q, FQ, R, dR, angle_uncertainty=0.002):
    r"""
    Returns the residuals from an error-in-variables model marginalized over
    the variables.

    **Warning** Assumes F(Q) is smoothly varying. If neighboring points have
    vastly different resolution then this assumption may not hold.

    *angular_uncertainty* from motor jitter in degrees, 1-$\sigma$.

    For error in variables fits with normal uncertainty, start with the
    following model:

    .. math::

        x &=& x_o + \epsilon_1 \text{for} \epsilon_1 ~ N(0, \Delta x^2) \\
        y &=& f(x) + \epsilon_2 \text{for} \epsilon_2 ~ N(0, \Delta y^2) \\

    Use a linear approximation at the nominal measument location $x_o$ then

    .. math::

        f(x) \approx f'(x_ο)(x - x_o) + f(x_o)

    and so

    .. math::

        y &\approx& f'(x_o)(x_o + \epsilon_1 - x_o +f(x_o) + \epsilon_2 \\
          &=& f(x_o) + f'(x_o)\epsilon_1 + \epsilon_2

    Therefore, measured $y_o$ is distributed as

    .. math::

        y_o &~& f(x_o) + f'(x_o)N(0, \Delta x^2) + N(0, \Delta y^2) \\
            &~& N(f(x_o), [f'(x_o)\Delta x]^2 + \Delta y^2)

    That is, assuming that f(x) is approximately linear over $\Delta x$, then
    simply add $[f'(x_o)\Delta x]^2$ to the variance in the data. Furthermore,
    assuming that we are sampling $x$ densely enough, then we can approximate
    $f'(x)$ using the center point formula

    .. math::

        f'(x) \approx \frac{f(x_{k+1}) - f(x_{k-1})}{x_{k+1} - x_{k-1}}

    For reflectometry specifically the motor uncertainty is in angle and the
    theory is in Q, so we use the following

    .. math::

        Q &=& \frac{4 \pi}{\lambda} \sin(\theta + \delta\theta) \\
          &=& \frac{4 \pi}{\lambda} (\cos \delta\theta \sin \theta
              + \sin \delta\theta \cos \theta)

    Using the small angle approximation $\delta\theta$ and $\cos \theta > 0.96$
    for $\theta < 15^o$, then

    .. math::

        Q &\approx& \frac{4 \pi}{\lambda} (\sin \theta + \delta\theta \cos \theta)
          &\approx& \frac{4 \pi}{\lambda} (\sin \theta + \delta\theta)
          &=$ Q + \frac{4 \pi}{\lambda}\delta\theta

    and so

    .. math::

        $\epsilon_1 = \delta Q = \frac{4 \pi}{\lambda}\delta\theta

    """
    # slope from center point formula
    if angle_uncertainty == 0.0:
        return (R - FQ) / dR
    # Using small angle approximation to Q = 4 pi/L sin(T + d)
    #    Q = 4 pi / L (cos d sin T + sin d cos T)
    #      ~ 4 pi / L (sin T + d cos T)    since d is small
    #      ~ 4 pi / L (sin T + d)          since cos T > 0.96 for T < 15 degrees
    #      = Q + 4 pi / L d
    #      ~ Q + 2.5 d                     since L in [4, 6] angstroms
    DQ = 2.5 * np.radians(angle_uncertainty)

    # Quick approx to [ log integral P(R,dR;Q') P(Q') dQ'] for motor position
    # uncertainty P(Q') and gaussian measurement uncertainty P(R;Q') is to
    # increase the size of dR based on the slope at Q and the motor uncertainty.
    dRdQ = (R[2:] - R[:-2]) / (Q[2:] - Q[:-2])
    # Leave dR untouched for the initial and final point
    dRp = dR.copy()
    dRp[1:-1] = np.sqrt((dR[1:-1]) ** 2 + (DQ * dRdQ) ** 2)  # add in quadrature
    return (R - FQ) / (dRp)


class DQExperiment(Experiment):
    def residuals(self):
        if "residuals" in self._cache:
            return self._cache["residuals"]

        if self.probe.polarized:
            have_data = not all(x is None or x.R is None for x in self.probe.xs)
        else:
            have_data = self.probe.R is not None
        if not have_data:
            resid = np.zeros(0)
            self._cache["residuals"] = resid
            return resid

        QR = self.reflectivity()
        if self.probe.polarized:
            resid = np.hstack(
                [
                    marginalized_residuals(QRi[0], QRi[1], xs.R, xs.dR, getattr(xs, "angle_uncertainty", 0.0))
                    for xs, QRi in zip(self.probe.xs, QR)
                    if xs is not None
                ]
            )
        else:
            resid = marginalized_residuals(
                QR[0], QR[1], self.probe.R, self.probe.dR, getattr(probe, "angle_uncertainty", 0.0)
            )
        self._cache["residuals"] = resid
        return resid


# Monkey-patch the residuals plotter
from bumps.plotutil import coordinated_colors, auto_shift


def plot_residuals(self, theory=None, suffix="", label=None, plot_shift=None, **kwargs):
    import matplotlib.pyplot as plt

    plot_shift = plot_shift if plot_shift is not None else Probe.residuals_shift
    trans = auto_shift(plot_shift)
    if theory is not None and self.R is not None:
        c = coordinated_colors()
        Q, R = theory
        # In case theory curve is evaluated at more/different points...
        R = np.interp(self.Q, Q, R)
        angle_uncertainty = getattr(self, "angle_uncertainty", 0.0)
        residual = marginalized_residuals(self.Q, R, self.R, self.dR, angle_uncertainty)
        plt.plot(
            self.Q, residual, ".", color=c["light"], transform=trans, label=self.label(prefix=label, suffix=suffix)
        )
    plt.axhline(1, color="black", ls="--", lw=1)
    plt.axhline(0, color="black", lw=1)
    plt.axhline(-1, color="black", ls="--", lw=1)
    plt.xlabel("Q (inv A)")
    plt.ylabel("(theory-data)/error")
    plt.legend(numpoints=1)


Probe.plot_residuals = plot_residuals

# ============ end of error-in-variables code =========

# Simulate some data
from bumps.util import push_seed

with push_seed(42):  # Repeatable data generation
    if angle_distribution == "uniform":
        # uniform motor uncertainty in [-angle_uncertainty, +angle_uncertainty]
        Toffset = (np.random.rand(len(T)) * 2 - 1) * angle_uncertainty
        angle_uncertainty = angle_uncertainty / np.sqrt(3)  # uniform => gaussian
    else:
        # gaussian motor uncertainty with 1-sigma angle_uncertainty
        Toffset = np.random.randn(len(T)) * angle_uncertainty

    # Marginalized residuals are computed assuming wavelength of 5 A.
    # Scale by 5/L if your average wavelength is significantly different.
    # TODO: use ratio of angle uncertainty and wavelength as control parameter
    angle_uncertainty = angle_uncertainty * 5 / L

    sim_probe = NeutronProbe(T=T + Toffset, dT=dT, L=L, dL=dL)
    sim_M = Experiment(probe=sim_probe, sample=sample)
    sim_M.simulate_data(noise=refl_error)
    R, dR = sim_probe.R, sim_probe.dR
    # print("angle shift", Toffset)
    # print(sim_probe.Q.shape, sim_probe.R.shape, sim_probe.dR.shape)
    # print("sim Q R dR", np.vstack((sim_probe.Q, sim_probe.R, sim_probe.dR)).T)

# Define the experiment
probe = NeutronProbe(T=T, dT=dT, L=L, dL=dL, data=(R, dR), name="eiv sim")
probe.angle_uncertainty = angle_uncertainty

if fit_strategy == "nominal":
    M = Experiment(probe=probe, sample=sample)
else:
    M = DQExperiment(probe=probe, sample=sample)
problem = FitProblem(M)
