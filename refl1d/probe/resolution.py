"""
Resolution calculations
"""

from typing import TYPE_CHECKING

from numpy import (
    arange,
    asarray,
    ceil,
    cos,
    degrees,
    float64,
    hstack,
    isscalar,
    log,
    ones_like,
    pi,
    radians,
    sin,
    sqrt,
    tan,
)
from numpy import arcsin as asin

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def QL2T(Q=None, L=None):
    r"""
    Compute angle from $Q$ and wavelength.

    .. math::

        \theta = \sin^{-1}( |Q| \lambda / 4 \pi )

    Returns $\theta$\ |deg|.
    """
    Q, L = asarray(Q, "d"), asarray(L, "d")
    return degrees(asin(abs(Q) * L / (4 * pi)))


def QT2L(Q=None, T=None):
    r"""
    Compute wavelength from $Q$ and angle.

    .. math::

        \lambda = 4 \pi \sin( \theta )/Q

    Returns $\lambda$\ |Ang|.
    """
    Q, T = asarray(Q, "d"), radians(asarray(T, "d"))
    return 4 * pi * sin(T) / Q


def TL2Q(T: "ArrayLike", L: "ArrayLike"):
    r"""
    Compute $Q$ from angle and wavelength.

    .. math::

        Q = 4 \pi \sin(\theta) / \lambda

    Returns $Q$ |1/Ang|
    """
    T, L = radians(asarray(T, "d")), asarray(L, "d")
    return 4 * pi * sin(T) / L


_FWHM_scale: float64 = sqrt(log(256))


def FWHM2sigma(s: "ArrayLike"):
    return asarray(s, "d") / _FWHM_scale


def sigma2FWHM(s):
    return asarray(s, "d") * _FWHM_scale


def dTdL2dQ(T: "ArrayLike", dT: "ArrayLike", L: "ArrayLike", dL: "ArrayLike"):
    r"""
    Convert wavelength dispersion and angular divergence to $Q$ resolution.

    *T*, *dT*  (degrees) angle and FWHM angular divergence
    *L*, *dL*  (Angstroms) wavelength and FWHM wavelength dispersion

    Returns 1-\ $\sigma$ $\Delta Q$

    Given $Q = 4 \pi sin(\theta)/\lambda$, this follows directly from
    gaussian error propagation using

    ..math::

        \Delta Q^2
            &= \left(\frac{\partial Q}{\partial \lambda}\right)^2\Delta\lambda^2
            + \left(\frac{\partial Q}{\partial \theta}\right)^2\Delta\theta^2

            &= Q^2 \left(\frac{\Delta \lambda}{\lambda}\right)^2
            + Q^2 \left(\frac{\Delta \theta}{\tan \theta}\right)^2

            &= Q^2 \left(\frac{\Delta \lambda}{\lambda}\right)^2
            + \left(\frac{4\pi\cos\theta\,\Delta\theta}{\lambda}\right)^2

    with the final form chosen to avoid cancellation at $Q=0$.
    """

    # Compute dQ from wavelength dispersion (dL) and angular divergence (dT)
    T, dT = radians(asarray(T, float64)), radians(asarray(dT, float64))
    L, dL = asarray(L, float64), asarray(dL, float64)
    # print T, dT, L, dL
    dQ = (4 * pi / L) * sqrt((sin(T) * dL / L) ** 2 + (cos(T) * dT) ** 2)

    # sqrt((dL/L)**2+(radians(dT)/tan(radians(T)))**2)*probe.Q
    return FWHM2sigma(dQ)


def dQ_broadening(dQ, L, T, dT, width):
    r"""
    Broaden an existing dQ by the given divergence.

    *dQ* |1/Ang|, with 1-\ $\sigma$ $Q$ resolution
    *L* |Ang|
    *T*, *dT* |deg|, with FWHM angular divergence
    *width* |deg|, with FWHM increased angular divergence

    The calculation is derived by substituting
    $\Delta\theta' = \Delta\theta + \omega$ for sample broadening $\omega$
    into the resolution estimate
    $(\Delta Q/Q)^2 = (\Delta\lambda/\lambda)^2 + (\Delta\theta/\tan\theta)^2$.
    """
    T, dT = radians(asarray(T, "d")), FWHM2sigma(radians(asarray(dT, "d")))
    width = FWHM2sigma(radians(width))
    dQsq = dQ**2 + (4 * pi / L * cos(T)) ** 2 * (2 * width * dT + width**2)

    # If width < -dT, need to take abs(dQsq) before taking the sqrt
    # (focusing past zero)
    return sqrt(abs(dQsq))


def dQdT2dLoL(Q, dQ, T, dT):
    r"""
    Convert a calculated Q resolution and angular divergence to a
    wavelength dispersion.

    *Q*, *dQ* |1/Ang|  $Q$ and 1-\ $\sigma$ $Q$ resolution
    *T*, *dT* |deg| angle and FWHM angular divergence

    Returns FWHM $\Delta\lambda/\lambda$
    """
    T, dT = radians(asarray(T, "d")), radians(asarray(dT, "d"))
    Q, dQ = asarray(Q, "d"), asarray(dQ, "d")
    dQoQ = sigma2FWHM(dQ) / Q
    dToT = dT / tan(T)
    if (dQoQ < dToT).any():
        raise ValueError("Cannot infer wavelength resolution: dQ is too small or dT is too large for some data points")
    return sqrt(dQoQ**2 - dToT**2)


def dQdL2dT(Q, dQ, L, dL):
    r"""
    Convert a calculated Q resolution and wavelength dispersion to
    angular divergence.

    *Q*, *dQ* |1/Ang|  $Q$ and 1-\ $\sigma$ $Q$ resolution
    *L*, *dL* |deg| angle and FWHM angular divergence

    Returns FWHM \Delta\theta$
    """
    L, dL = asarray(L, "d"), asarray(dL, "d")
    Q, dQ = asarray(Q, "d"), asarray(dQ, "d")
    T = radians(QL2T(Q, L))
    dQoQ = sigma2FWHM(dQ) / Q
    dLoL = dL / L
    if (dQoQ < dLoL).any():
        raise ValueError("Cannot infer angular resolution: dQ is too small or dL is too large for some data points")
    dT = degrees(sqrt(dQoQ**2 - dLoL**2) * tan(T))
    return dT


Plancks_constant = 6.62618e-27  # Planck constant (erg*sec)
neutron_mass = 1.67495e-24  # neutron mass (g)


def TOF2L(d_moderator, TOF):
    r"""
    Convert neutron time-of-flight to wavelength.

    .. math::

        \lambda = (t/d) (h/n_m)

    where:

        | $\lambda$ is wavelength in |Ang|
        | $t$ is time-of-flight in $u$\s
        | $h$ is Planck's constant in erg seconds
        | $n_m$ is the neutron mass in g
    """
    return TOF * (Plancks_constant / neutron_mass / d_moderator)


def bins(low, high, dLoL):
    r"""
    Return bin centers from low to high preserving a fixed resolution.

    *low*, *high* are the minimum and maximum wavelength.
    *dLoL* is the desired resolution FWHM $\Delta\lambda/\lambda$ for the bins.
    """

    step = 1 + dLoL
    n = ceil(log(high / low) / log(step))
    edges = low * step ** arange(n + 1)
    L = (edges[:-1] + edges[1:]) / 2
    return L


def binwidths(L):
    r"""
    Determine the wavelength dispersion from bin centers *L*.

    The wavelength dispersion $\Delta\lambda$ is just the difference
    between consecutive bin edges, so:

    .. math::

        \Delta L_i  = E_{i+1}-E_{i}
                    = (1+\omega) E_i - E_i
                    = \omega E_i
                    = \frac{2 \omega}{2+\omega} L_i

    where $E$ and $\omega$ are as defined in :func:`binedges`.
    """
    L = asarray(L, "d")
    if L[1] > L[0]:
        dLoL = L[1] / L[0] - 1
    else:
        dLoL = L[0] / L[1] - 1
    dL = 2 * dLoL / (2 + dLoL) * L
    return dL


def binedges(L):
    r"""
    Construct bin edges *E* from bin centers *L*.

    Assuming fixed $\omega = \Delta\lambda/\lambda$ in the bins, the
    edges will be spaced logarithmically at:

    .. math::

        E_0     &= \min \lambda \\
        E_{i+1} &= E_i + \omega E_i = E_i (1+\omega)

    with centers $L$ half way between the edges:

    .. math::

        L_i = (E_i+E_{i+1})/2
            = (E_i + E_i (1+\omega))/2
            = E_i (2 + \omega)/2

    Solving for $E_i$, we can recover the edges from the centers:

    .. math::

        E_i = L_i \frac{2}{2+\omega}

    The final edge, $E_{n+1}$, does not have a corresponding center
    $L_{n+1}$ so we must determine it from the previous edge $E_n$:

    .. math::

        E_{n+1} = L_n \frac{2}{2+\omega}(1+\omega)

    The fixed $\omega$ can be retrieved from the ratio of any pair
    of bin centers using:

    .. math::

        \frac{L_{i+1}}{L_i} = \frac{ (E_{i+2}+E_{i+1})/2 }{ (E_{i+1}+E_i)/2 }
                          = \frac{ (E_{i+1}(1+\omega)+E_{i+1} }
                                  { (E_i(1+\omega)+E_i }
                          = \frac{E_{i+1}}{E_i}
                          = \frac{E_i(1+\omega)}{E_i} = 1 + \omega
    """
    L = asarray(L, "d")
    if L[1] > L[0]:
        dLoL = L[1] / L[0] - 1
        last = 1 + dLoL
    else:
        dLoL = L[0] / L[1] - 1
        last = 1.0 / (1 + dLoL)
    E = L * 2 / (2 + dLoL)
    return hstack((E, E[-1] * last))


def divergence(T=None, slits=None, distance=None, sample_width=1e10, sample_broadening=0):
    r"""
    Calculate divergence due to slit and sample geometry.

    :Parameters:
        *T*         : float OR [float] | degrees
            incident angles
        *slits*     : float OR (float, float) | mm
            s1, s2 slit openings for slit 1 and slit 2
        *distance*  : (float, float) | mm
            d1, d2 distance from sample to slit 1 and slit 2
        *sample_width*      : float | mm
            w, width of the sample
        *sample_broadening* : float | degrees FWHM
            additional divergence caused by sample

    :Returns:
        *dT*  : float OR [float] | degrees FWHM
            calculated angular divergence

    **Algorithm:**

    The divergence is based on the slit openings and the distance between
    the slits.  For very small samples, where the slit opening is larger
    than the width of the sample across the beam, the sample itself acts
    like the second slit.

    First find $p$, the projection of the beam on the sample:

    .. math::

        p &= w \sin\left(\frac{\pi}{180}\theta\right)

    Depending on whether $p$ is larger than $s_2$, determine the slit
    divergence $\Delta\theta_d$ in radians:

    .. math::

        \Delta\theta_d &= \left\{
          \begin{array}{ll}
            \frac{1}{2}\frac{s_1+s_2}{d_1-d_2} & \mbox{if } p \geq s_2 \\
            \frac{1}{2}\frac{s_1+p}{d_1}       & \mbox{if } p < s_2
          \end{array}
        \right.

    In addition to the slit divergence, we need to add in any sample
    broadening $\Delta\theta_s$ returning the total divergence in degrees:

    .. math::

        \Delta\theta &= \frac{180}{\pi} \Delta\theta_d + \Delta\theta_s

    Reversing this equation, the sample broadening contribution can
    be measured from the full width at half maximum of the rocking
    curve, $B$, measured in degrees at a particular angle and slit
    opening:

    .. math::

        \Delta\theta_s = B - \frac{180}{\pi}\Delta\theta_d
    """
    # TODO: update from reductus to handle four slits
    # TODO: check that the formula is correct for T=0 => dT = s1 / d1
    # TODO: add sample_offset and compute full footprint
    d1, d2 = distance
    try:
        s1, s2 = slits
    except TypeError:
        s1 = s2 = slits

    # Compute FWHM angular divergence dT from the slits in degrees
    dT = degrees(0.5 * (s1 + s2) / (d1 - d2))

    # For small samples, use the sample projection instead.
    sample_s = sample_width * sin(radians(T))
    if isscalar(sample_s):
        if sample_s < s2:
            dT = degrees(0.5 * (s1 + sample_s) / d1)
    else:
        idx = sample_s < s2
        # print s1, s2, d1, d2, T, dT, sample_s
        s1 = ones_like(sample_s) * s1
        dT = ones_like(sample_s) * dT
        dT[idx] = degrees(0.5 * (s1[idx] + sample_s[idx]) / d1)

    return dT + sample_broadening


def slit_widths(T=None, slits_at_Tlo=None, Tlo=90, Thi=90, slits_below=None, slits_above=None):
    """
    Compute the slit widths for the standard scanning reflectometer
    fixed-opening-fixed geometry.

    :Parameters:
        *T* : [float] | degrees
            Specular measurement angles.
        *Tlo*, *Thi* : float | degrees
            Start and end of the opening region.  The default if *Tlo* is
            not specified is to use fixed slits at *slits_below* for all
            angles.
        *slits_below*, *slits_above* : float OR [float, float] | mm
            Slits outside opening region.  The default is to use the
            values of the slits at the ends of the opening region.
        *slits_at_Tlo* : float OR [float, float] | mm
            Slits at the start of the opening region.

    :Returns:
        *s1*, *s2* : [float] | mm
            Slit widths for each theta.

    Slits are assumed to be fixed below angle *Tlo* and above angle *Thi*,
    and opening at a constant dT/T between them.

    Slit openings are defined by a tuple (s1, s2) or constant s=s1=s2.
    With no *Tlo*, the slits are fixed with widths defined by *slits_below*,
    which defaults to *slits_at_Tlo*.  With no *Thi*, slits are continuously
    opening above *Tlo*.

    .. Note::
         This function works equally well if angles are measured in
         radians and/or slits are measured in inches.

    """

    # Slits at T<Tlo
    if slits_below is None:
        slits_below = slits_at_Tlo
    try:
        b1, b2 = slits_below
    except TypeError:
        b1 = b2 = slits_below
    s1 = ones_like(T) * b1
    s2 = ones_like(T) * b2

    # Slits at Tlo<=T<=Thi
    try:
        m1, m2 = slits_at_Tlo
    except TypeError:
        m1 = m2 = slits_at_Tlo
    idx = abs(T) >= Tlo
    s1[idx] = m1 * T[idx] / Tlo
    s2[idx] = m2 * T[idx] / Tlo

    # Slits at T > Thi
    if slits_above is None:
        slits_above = m1 * Thi / Tlo, m2 * Thi / Tlo
    try:
        t1, t2 = slits_above
    except TypeError:
        t1 = t2 = slits_above
    idx = abs(T) > Thi
    s1[idx] = t1
    s2[idx] = t2

    return s1, s2


'''
def resolution(Q=None, s=None, d=None, L=None, dLoL=None, Tlo=None, Thi=None,
               s_below=None, s_above=None,
               broadening=0, sample_width=1e10, sample_distance=0):
    """
    Compute the resolution for Q on scanning reflectometers.

    broadening is the sample warp contribution to angular divergence, as
    measured by a rocking curve.  The value should be w - (s1+s2)/(2*d)
    where w is the full-width at half maximum of the rocking curve.

    For itty-bitty samples, provide a sample width w and sample distance ds
    from slit 2 to the sample.  If s_sample = sin(T)*w is smaller than s2
    for some T, then that will be used for the calculation of dT instead.

    """
    T = QL2T(Q=Q, L=L)
    slits = slit_widths(T=T, s=s, Tlo=Tlo, Thi=Thi)
    dT = divergence(T=T, slits=slits, sample_width=sample_width,
                    sample_distance=sample_distance) + broadening
    Q, dQ = Qresolution(L, dLoL*L, T, dT)
    return FWHM2sigma(dQ)


def demo():
    from numpy import linspace, exp, real, conj, sin, radians
    import matplotlib.pyplot as plt

    # Values from volfrac example in garefl
    T = linspace(0, 9, 140)
    Q = 4*pi*sin(radians(T))/5.0042
    dQ = resolution(Q, s=0.21, Tlo=0.35, d=1890., L=5.0042, dLoL=0.009)
    #plt.plot(Q, dQ)

    # Fresnel reflectivity for silicon
    rho, sigma=2.07, 5
    kz=Q/2
    f = sqrt(kz**2 - 4*pi*rho*1e-6 + 0j)
    r = (kz-f)/(kz+f)*exp(-2*sigma**2*kz*f)
    r[abs(kz)<1e-10] = -1
    R = real(r*conj(r))
    plt.errorbar(Q, R, xerr=dQ, fmt=',r', capsize=0)
    plt.grid(True)
    plt.semilogy(Q, R, ',b')

    plt.show()


def demo2():
    import numpy as np
    import matplotlib.pyplot as plt

    Q, R, dR = np.loadtxt('ga128.refl.mce').T
    dQ = resolution(Q, s=0.154, Tlo=0.36, d=1500., L=4.75, dLoL=0.02)
    plt.errorbar(Q, R, xerr=dQ, yerr=dR, fmt=',r', capsize=0)
    plt.grid(True)
    plt.semilogy(Q, R, ',b')
    plt.show()


if __name__ == "__main__":
    demo2()
'''
