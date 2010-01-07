# This program is in the public domain
# Author: Paul Kienzle
"""
Functions to compute resolution.

Resolution is calculated from the angular divergence (which depends on slits,
and therefore depends on angle as slits open with angle) and wavelength spread
(which is constant for scanning instruments, but varying on TOF instruments).
Normally data is taken with fixed slits at low angles until the beam falls
onto the sample, then the slits are opened gradually with angle so that the
beam stays on the sample (increasing intensity at the cost of worse
resolution), until they reach some maximum beyond which they are held
fixed again.

For itty-bitty samples, the width of the sample as seen by the beam is smaller
than the second slit, and so the resolution should be computed from that
instead.  This isn't quite correct for very low Q, but in that case both
reflected and transmitted neutrons will arrive at the detector, so the
computed resolution of dQ=0 at Q=0 is good enough.

An additional factor to resolution is that the sample may be warped.  If it
bows outward, this is a constant increase in the wavelength divergence.  If
it bows inward, this will be a decrease.  A second order effect is that at
low angles the warping will cast shadows, changing the resolution and
intensity in very complex ways.

Resolution in Q is computed from uncertainty in wavelength L and angle T
using propogation of errors::

    dQ**2 = (df/dL)**2 dL**2 + (df/dT)**2 dT**2

where::

    f(L,T) = 4 pi sin(T)/L
    df/dL = -4 pi sin(T)/L**2 = -Q/L
    df/dT = 4 pi cos(T)/L = (4 pi sin(T) / L) / (sin(T)/cos(T)) = Q/tan(T)

yielding the traditional form::

    (dQ/Q)**2 = (dL/L)**2 + (dT/tan(T))**2

Computationally, 1/tan(T) blows up at 0, so it is better to use the
direct calculation::

    dQ = (4 pi / L) sqrt( sin(T)**2 (dL/L)**2 + cos(T)**2 dT**2 )


Wavelength dispersion dL/L is constant (e.g., for AND/R it is 2% FWHM).  It
can vary on time-of-flight instruments.  FWHM can be converted to 1-sigma
resolution using the scale factor of 1/sqrt(8 * log(2))

Angular divergence dT comes primarily from the slit geometry, but can have
broadening or focusing due to a warped sample.  The slit contribution is
dT = (s1+s2)/(2d) FWHM  where s1,s2 are slit openings and d is the distance
between slits.  For itty-bitty samples, the sample itself can act as a
slit.  If s_sample = sin(T)*w is smaller than s2 for some T, then use
dT = (s1+s_sample)/(2(d+d_sample)) instead.

The sample broadening can be read off a rocking curve as  w - (s1+s2)/(2d)
where w is the measured FWHM of the peak. This constant should be added to
the computed dT for all angles and slit geometries. You will not
usually have this information on hand, but you can leave space for users
to enter it if it is available.

For opening slits, dT/T is held constant, so if you know s and To at the
start of the opening slits region you can compute dT/To, and later scale
that to your particular T::

    dT(Q) = dT/To * T(Q)

Because d is fixed, that means s1(T) = s1(To) * T/To and s2(T) = s2(To) * T/To
"""
from numpy import pi, ones_like, sqrt, log, degrees, radians, cos, sin
from numpy import arcsin as asin


def QL2T(Q=None,L=None):
    """
    Compute angle from Q and wavelength.
    """
    return degrees(asin(abs(Q) * L / (4*pi)))

def slits_from_angle(T=None,slits_at_Tlo=None,Tlo=None,Thi=None,
                     slits_below=None, slits_above=None):
    """
    Compute the slit openings for the standard scanning reflectometer
    fixed-opening-fixed geometry.

    Slits are assumed to be fixed below angle *Tlo* and above angle *Thi*,
    and opening at a constant dT/T between them.

    Slit openings are recorded at *Tlo* as a tuple (s1,s2) or constant s=s1=s2.
    *Tlo* is optional for completely fixed slits.  *Thi* is optional if there
    is no top limit to the fixed slits.

    *slits_below* are the slits at *T* < *Tlo*.
    *slits_above* are the slits at *T* > *Thi*.
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
    if Tlo != None:
        try:
            m1,m2 = slits_at_Tlo
        except TypeError:
            m1=m2 = slits_at_Tlo
        idx = (abs(T) >= Tlo) & (abs(T) <= Thi)
        s1[idx] = m1 * T[idx]/Tlo
        s2[idx] = m2 * T[idx]/Tlo

    # Slits at T > Thi
    if Thi != None:
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

def divergence(T=None, slits=None, sample_width=1e10, d_s1=None, d_s2=None):
    """
    Calculate divergence due to slit and sample geometry.

    Note: default sample width is large but not infinite so that at Q=0,
    sin(0)*sample_width returns 0 rather than NaN.  This does lead to a
    resolution dQ=0 at Q=0, but that shouldn't be a problem.
    """
    # TODO: add sample_offset and compute full footprint
    s1,s2 = slits

    # Compute angular divergence dT from the slits
    dT = (s1+s2)/(2*(d_s1-d_s2))

    # For small samples, use the sample projection instead.
    sample_s = sample_width * sin(radians(T))
    idx = sample_s < s2
    dT[idx] = (s1[idx] + sample_s[idx])/(2*d_s1)

    return dT

def delta_Q(L, dLoL, T, dT):

    # Compute dQ from wavelength dispersion (dL) and angular divergence (dT)
    T,dT = radians(T), radians(dT)
    dQ = (4*pi/L) * sqrt( (sin(T)*dLoL)**2 + (cos(T)*dT)**2 )

    # Return dQ as sigma rather than FWHM
    return dQ / sqrt(log(256))

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
    slits = opening_slits(T=T, s=s, Tlo=Tlo, Thi=Thi)
    dT = divergence(T=T,slits=slits, sample_width=sample_width,
                    sample_distance=sample_distance) + broadening
    dQ = delta_Q(L, dLoL, T, dT)
    return dQ


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
