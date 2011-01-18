r"""
Resolution calculations

Given $Q = 4 \pi \sin(\theta)/\lambda$, the instrumental resolution in $Q$ is
determined by the dispersion angle $\theta$ and wavelength $\lambda$.  For
monochromatic instruments, the wavelength resolution is fixed and the
angular resolution varies.  For polychromatic instruments, the wavelength
resolution varies and the angular resolution is fixed for each measurement.

The angular resolution is determined by the geometry (slit positions,
openings and sample profile) with perhaps an additional contribution
from sample warp.  For monochromatic instruments, measurements are taken
with fixed slits at low angles until the beam falls completely onto the
sample.  Then as the angle increases, slits are opened to preserve full
illumination.  At some point the slit openings exceed the beam width,
and thus they are left fixed for all angles above this threshold.

When the sample is tiny, stray neutrons miss the sample and are not
reflected onto the detector.  This results in a resolution that is
tighter than expected given the slit openings.  If the sample width
is available, we can use that to determine how much of the beam is
intercepted by the sample, which we then use as an alternative second
slit.  This simple calculation isn't quite correct for very low Q, but
data in this region will be contaminated by the direct beam, so we
won't be using those points.

When the sample is warped, it may act to either focus or spread the
incident beam.  Some samples are diffuse scatters, which also acts
to spread the beam.  The degree of spread can be estimated from the
full-width at half max (FWHM) of a rocking curve at known slit settings.
The expected FWHM will be $1/2 (s_1+s_2)/(d_1-d_2)$.  The difference
between this and the measured FWHM is the sample_broadening value.
A second order effect is that at low angles the warping will cast
shadows, changing the resolution and intensity in very complex ways.

For polychromatic time of flight instruments, the wavelength dispersion
is determined by the reduction process which usually bins the time
channels in a way that sets a fixed relative resolution
$\Delta \lambda / \lambda$ for each bin.



Algorithms
==========

Resolution in Q is computed from uncertainty in wavelength $\sigma_\lambda$
and angle $\sigma_\theta$ using propagation of errors:

..math::

    \sigma^2_Q
        &= \left|\frac{\partial Q}{\partial \lambda}\right|^2 \sigma_\lambda^2
         + \left|\frac{\partial Q}{\partial \theta}\right|^2 \sigma_\theta^2
         + 2 \left|\frac{\partial Q}{\partial \lambda}
                   \frac{\partial Q}{\partial \theta}\right|^2
                   \sigma_{\lambda\theta}
         \\
    Q &= 4 \pi \sin(\theta) / \lambda \\
    \frac{\partial Q}{\partial \lambda} &= -4 \pi \sin(\theta)/\lambda^2
         = -Q/\lambda \\
    \frac{\partial Q}{\partial \theta} &= 4 pi \cos(\theta)/\lambda
         = \cos(\theta) \cdot Q/\sin(\theta) = Q/\tan(\theta)

With no correlation between wavelength dispersion and angular divergence,
$\sigma_{\theta\lambda} = 0$, yielding the traditional form:

..math::

    \left(\frac{\Delta Q}{Q}\right)^2
         = \left(\frac{\Delta \lambda}{\lambda}\right)^2
         + \left(\frac{\Delta \theta}{\theta}\right)^2

Computationally, $1/\tan(\theta)$ is infinity at $\theta=0$, so it is better
to use the direct calculation:

..math::

    \Delta Q = 4 \pi/\lambda \sqrt{\sin(\theta)^2 (\Delta\lambda/\lambda)^2
                                   + \cos(\theta)^2 \Delta \theta^2}

Wavelength dispersion $\Delta \lambda/\lambda$ is usually constant
(e.g., for AND/R it is 2% FWHM), but it can vary on time-of-flight
instruments depending on how the data is binned.

Angular divergence $\delta \theta$ comes primarily from the slit geometry,
but can have broadening or focusing due to a warped sample.  The FWHM
divergence in radians due to slits is:

..math::

    \Delta\theta_{\rm slits} = \frac{1}{2} \frac{s_1 + s_2}{d_1 - d_2}

where $s_1,s_2$ are slit openings edge to edge and $d_1,d_2$ are the distances
between the sample and the slits.  For tiny samples of width $m$, the sample
itself can act as a slit.  If $s = m \sin(\theta)$ is smaller than $s_2$ for
some $\theta$, then use:

..math::

    \Delta\theta_{\rm slits} = \frac{1}{2} \frac{s_1 + m \sin(\theta)}{d_1}

The sample broadening can be read off a rocking curve using:

..math::

    \Delta\theta_{\rm sample} = w - \Delta\theta_{\rm slits}

where $w$ is the measured FWHM of the peak in degrees. Broadening can be
negative for concave samples which have a focusing effect on the beam.  This
constant should be added to the computed $\Delta \theta$ for all angles and
slit geometries.  You will not usually have this information on hand, but
you can leave space for users to enter it if it is available.

FWHM can be converted to 1-\ $\sigma$ resolution using the scale factor of
$1/\sqrt{8 \ln 2}$.

With opening slits we assume $\Delta \theta/\theta$ is held constant, so if
you know $s$ and $\theta_o$ at the start of the opening slits region you
can compute $\Delta \theta/\theta_o$, and later scale that to your
particular $\theta$:

..math::

    \Delta\theta(Q) = \Delta\theta/\theta_o \cdot \theta(Q)

Because $d$ is fixed, that means
$s_1(\theta) = s_1(\theta_o) \cdot \theta/\theta_o$ and
$s_2(\theta) = s_2(\theta_o) \cdot \theta/\theta_o$.
"""

from numpy import pi, inf, nan, sqrt, log, degrees, radians, cos, sin, tan
from numpy import arcsin as asin, ceil, clip
from numpy import ones_like, arange, isscalar, asarray, hstack

def QL2T(Q=None,L=None):
    """
    Compute angle from $Q$ and wavelength.

    .. math::

        \theta = \sin^{-1}( |Q| \lambda / 4 \pi )

    Returns $\theta$\ |deg|.
    """
    return degrees(asin(abs(Q) * L / (4*pi)))

def TL2Q(T=None,L=None):
    """
    Compute $Q$ from angle and wavelength.

    .. math::
    
        Q = 4 \pi \sin(\theta) / \lambda

    Returns $Q$ |1/Ang|
    """
    return 4 * pi * sin(radians(T)) / L

_FWHM_scale = sqrt(log(256))
def FWHM2sigma(s):
    return s/_FWHM_scale
def sigma2FWHM(s):
    return s*_FWHM_scale


def dTdL2dQ(T=None, dT=None, L=None, dL=None):
    """
    Convert wavelength dispersion and angular divergence to $Q$ resolution.

    *T*,*dT*  (degrees) angle and FWHM angular divergence
    *L*,*dL*  (Angstroms) wavelength and FWHM wavelength dispersion

    Returns 1-\ $\sigma$ $\Delta Q$
    """

    # Compute dQ from wavelength dispersion (dL) and angular divergence (dT)
    T,dT = radians(T), radians(dT)
    #print T, dT, L, dL
    dQ = (4*pi/L) * sqrt( (sin(T)*dL/L)**2 + (cos(T)*dT)**2 )

    #sqrt((dL/L)**2+(radians(dT)/tan(radians(T)))**2)*probe.Q
    return FWHM2sigma(dQ)

def dQdT2dLoL(Q, dQ, T, dT):
    """
    Convert a calculated Q resolution and wavelength divergence to a
    wavelength dispersion.

    *Q*, *dQ* |1/Ang|  $Q$ and 1-\ $\sigma$ $Q$ resolution
    *T*, *dT* |deg| angle and FWHM angular divergence

    Returns FWHM $\Delta\lambda/\lambda$
    """
    return sqrt( (sigma2FWHM(dQ)/Q)**2 - (radians(dT)/tan(radians(T)))**2 )




def bins(low, high, dLoL):
    """
    Return bin centers from low to high preserving a fixed resolution.

    *low*, *high* are the minimum and maximum wavelength.
    *dLoL* is the desired resolution FWHM $\Delta\lambda/\lambda$ for the bins.
    """

    step = 1 + dLoL;
    n = ceil(log(high/low)/log(step))
    edges = low*step**arange(n+1)
    L = (edges[:-1]+edges[1:])/2
    return L

def binwidths(L):
    """
    Construct dL assuming that L represents the bin centers of a
    measured TOF data set, and dL is the bin width.

    The bins L are assumed to be spaced logarithmically with edges::

        E[0] = min wavelength
        E[i+1] = E[i] + dLoL*E[i]

    and centers::

        L[i] = (E[i]+E[i+1])/2
             = (E[i] + E[i]*(1+dLoL))/2
             = E[i]*(2 + dLoL)/2

    so::

        L[i+1]/L[i] = E[i+1]/E[i] = (1+dLoL)
        dL[i] = E[i+1]-E[i] = (1+dLoL)*E[i]-E[i]
              = dLoL*E[i] = 2*dLoL/(2+dLoL)*L[i]
    """
    if L[1] > L[0]:
        dLoL = L[1]/L[0] - 1
    else:
        dLoL = L[0]/L[1] - 1
    dL = 2*dLoL/(2+dLoL)*L
    return dL

def binedges(L):
    """
    Construct bin edges E assuming that L represents the bin centers of a
    measured TOF data set.

    The bins L are assumed to be spaced logarithmically with edges::

        E[0] = min wavelength
        E[i+1] = E[i] + dLoL*E[i]

    and centers::

        L[i] = (E[i]+E[i+1])/2
             = (E[i] + E[i]*(1+dLoL))/2
             = E[i]*(2 + dLoL)/2

    so::

        E[i] = L[i]*2/(2+dLoL)
        E[n+1] = L[n]*2/(2+dLoL)*(1+dLoL)
    """
    if L[1] > L[0]:
        dLoL = L[1]/L[0] - 1
        last = (1+dLoL)
    else:
        dLoL = L[0]/L[1] - 1
        last = 1./(1+dLoL)
    E = L*2/(2+dLoL)
    return hstack((E,E[-1]*last))

def divergence(T=None, slits=None, distance=None,
               sample_width=1e10, sample_broadening=0):
    """
    Calculate divergence due to slit and sample geometry.

    :Parameters:
        *T*         : float OR [float] | degrees
            incident angles
        *slits*     : float OR (float,float) | mm
            s1,s2 slit openings for slit 1 and slit 2
        *distance*  : (float,float) | mm
            d1,d2 distance from sample to slit 1 and slit 2
        *sample_width*      : float | mm
            w, width of the sample
        *sample_broadening* : float | degrees FWHM
            additional divergence caused by sample

    :Returns:
        *dT*  : float OR [float] | degrees FWHM
            calculated angular divergence

    **Algorithm:**

    Uses the following formula:

        p = w * sin(radians(T))
        dT = /  1/2 (s1+s2) / (d1-d2)   if p >= s2
             \  1/2 (s1+p) /  d1        otherwise
        dT = degrees(dT) + sample_broadening

    where p is the projection of the sample into the beam.

    *sample_broadening* can be estimated from W, the FWHM of a rocking curve::

        sample_broadening = W - degrees( 0.5*(s1+s2) / (d1-d2))

    .. Note::
        Default sample width is large but not infinite so that at T=0,
        sin(0)*sample_width returns 0 rather than NaN.

    """
    # TODO: check that the formula is correct for T=0 => dT = s1 / d1
    # TODO: add sample_offset and compute full footprint
    d1,d2 = distance
    try:
        s1,s2 = slits
    except TypeError:
        s1=s2 = slits

    # Compute FWHM angular divergence dT from the slits in degrees
    dT = degrees(0.5*(s1+s2)/(d1-d2))

    # For small samples, use the sample projection instead.
    sample_s = sample_width * sin(radians(T))
    if isscalar(sample_s):
        if sample_s < s2: dT = degrees(0.5*(s1+sample_s)/d1)
    else:
        idx = sample_s < s2
        #print s1,s2,d1,d2,T,dT,sample_s
        s1 = ones_like(sample_s)*s1
        dT = ones_like(sample_s)*dT
        dT[idx] = degrees(0.5*(s1[idx] + sample_s[idx])/d1)

    return dT + sample_broadening

def slit_widths(T=None,slits_at_Tlo=None,Tlo=90,Thi=90,
                  slits_below=None, slits_above=None):
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
        *slits_below*, *slits_above* : float OR [float,float] | mm
            Slits outside opening region.  The default is to use the
            values of the slits at the ends of the opening region.
        *slits_at_Tlo* : float OR [float,float] | mm
            Slits at the start of the opening region.

    :Returns:
        *s1*, *s2* : [float] | mm
            Slit widths for each theta.

    Slits are assumed to be fixed below angle *Tlo* and above angle *Thi*,
    and opening at a constant dT/T between them.

    Slit openings are defined by a tuple (s1,s2) or constant s=s1=s2.
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
        b1,b2 = slits_below
    except TypeError:
        b1=b2 = slits_below
    s1 = ones_like(T) * b1
    s2 = ones_like(T) * b2

    # Slits at Tlo<=T<=Thi
    try:
        m1,m2 = slits_at_Tlo
    except TypeError:
        m1=m2 = slits_at_Tlo
    idx = abs(T) >= Tlo
    s1[idx] = m1 * T[idx]/Tlo
    s2[idx] = m2 * T[idx]/Tlo

    # Slits at T > Thi
    if slits_above is None:
        slits_above = m1 * Thi/Tlo, m2 * Thi/Tlo
    try:
        t1,t2 = slits_above
    except TypeError:
        t1=t2 = slits_above
    idx = abs(T) > Thi
    s1[idx] = t1
    s2[idx] = t2

    return s1,s2


'''
def resolution(Q=None,s=None,d=None,L=None,dLoL=None,Tlo=None,Thi=None,
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
    T = QL2T(Q=Q,L=L)
    slits = slit_widths(T=T, s=s, Tlo=Tlo, Thi=Thi)
    dT = divergence(T=T,slits=slits, sample_width=sample_width,
                    sample_distance=sample_distance) + broadening
    Q,dQ = Qresolution(L, dLoL*L, T, dT)
    return FWHM2sigma(dQ)

def demo():
    import pylab
    from numpy import linspace, exp, real, conj, sin, radians
    # Values from volfrac example in garefl
    T = linspace(0,9,140)
    Q = 4*pi*sin(radians(T))/5.0042
    dQ = resolution(Q,s=0.21,Tlo=0.35,d=1890.,L=5.0042,dLoL=0.009)
    #pylab.plot(Q,dQ)

    # Fresnel reflectivity for silicon
    rho,sigma=2.07,5
    kz=Q/2
    f = sqrt(kz**2 - 4*pi*rho*1e-6 + 0j)
    r = (kz-f)/(kz+f)*exp(-2*sigma**2*kz*f)
    r[abs(kz)<1e-10] = -1
    R = real(r*conj(r))
    pylab.errorbar(Q,R,xerr=dQ,fmt=',r',capsize=0)
    pylab.grid(True)
    pylab.semilogy(Q,R,',b')

    pylab.show()

def demo2():
    import numpy,pylab
    Q,R,dR = numpy.loadtxt('ga128.refl.mce').T
    dQ = resolution(Q, s=0.154, Tlo=0.36, d=1500., L=4.75, dLoL=0.02)
    pylab.errorbar(Q,R,xerr=dQ,yerr=dR,fmt=',r',capsize=0)
    pylab.grid(True)
    pylab.semilogy(Q,R,',b')
    pylab.show()



if __name__ == "__main__":
    demo2()
'''
