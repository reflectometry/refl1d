# This program is in the public domain
# Author: Paul Kienzle
"""
Experimental probe.

The experimental probe describes the incoming beam for the experiment.

Scattering properties of the sample are dependent on the type and
energy of the radiation.
"""

import numpy
from numpy import radians, sin, sqrt, tan, cos, pi, inf, sign
from periodictable import nsf, xsf
from calc import convolve
from . import fresnel
from material import Vacuum
from mystic.parameter import Parameter

def real_to_Q(angle=None, wavelength=None):
    Arad = radians(angle)
    # Q = 4pi/L sin(A)
    Q = 4 * pi / wavelength * sin(Arad)
    return Q

def Qresolution(angle, wavelength):
    A,dA = angle
    L,dL = wavelength
    Arad,dArad = radians(A), radians(dA)
    # Q = 4pi/L sin(A)
    # dQ**2 = 4pi/L sqrt( (dL/L)**2 sin(A)**2 +  cos(A)**2 dA**2)
    # (dQ/Q)**2 = sqrt((dL/L)**2 + (dA/tan(A))**2)
    Q = 4 * pi / L * sin(Arad)
    dQ = Q * sqrt((dL/L)**2 + (dArad/tan(Arad))**2)
    return Q,dQ

class Probe: # Abstract base class
    """
    Defines the incident beam used to study the material.

    The probe is used to compute the scattering potential for the individual
    materials that the beam will bass through.  This potential is normalized
    to density=1 g/cm**3.  To use these values in the calculation of
    reflectivity, they need to be scaled by density and volume fraction.

    The choice of normalized density is dictated by the cost of lookups
    in the CXRO scattering tables.  With normalized density only one lookup
    is necessary during the fit, rather than one for each choice of density
    even when density is not the fitting parameter.

    For calculation purposes, probe needs to return the values, Q_calc, at
    which the model is evaluated.  This is normally going to be the measured
    points only, but for some systems, such as those with very thick layers,
    oversampling is needed to avoid aliasing effects.

    Measurement properties::

        *intensity* is the beam intensity
        *background* is the background
        *back_absorption* is the amount of absorption through the substrate
        *theta_offset* is the offset of the sample from perfect alignment

    All measurement properties are fittable parameters.  *theta_offset* in
    particular should be set using probe.theta_offset.dev(dT), with dT
    equal to the uncertainty in the peak position for the rocking curve,
    as measured in radians.  Changes to *theta_offset* will then be penalized
    in the cost function for the fit as if it were another measurement.  Note
    that the uncertainty in the peak position is not the same as the width
    of the peak.  The peak stays roughly the same as statistics are improved, 
    but the uncertainty in position and width will decrease.

    View properties::

        *substrate* is the material which makes up the substrate
        *surround* is the material which makes up the surround
        *view* is 'fresnel', 'log', 'linear' or 'Q**4'

    Normally *view* is set directly in the class rather than the
    instance since it is not specific to the view.  The fresnel
    substrate and surround materials are a property of the sample,
    and should share the same material.
    """
    polarized = False
    view = "fresnel"
    substrate = None
    surround = None
    def __init__(self):
        self.intensity = Parameter.default(1,name="intensity")
        self.background = Parameter.default(0,name="background", 
                                            limits=[0,inf])
        self.back_absorption = Parameter.default(1, name="back_absorption",
                                                 limits=[0,1])
        self.theta_offset = Parameter.default(0,name="theta_offset")    
    def _Q(self):
        if self.theta_offset.value != 0:
            Q = real_to_Q(angle=self.A+self.theta_offset.value,
                          wavelength=self.L)
        else:
            Q = self.Qo
        return Q
    Q = property(_Q)
    def _calc_Q(self):
        if self.theta_offset.value != 0:
            Q = real_to_Q(angle=self.calc_A+self.theta_offset.value,
                          wavelength=self.calc_L)
        else:
            Q = self.calc_Qo
        return Q
    calc_Q = property(_calc_Q)
    def parameters(self):
        return dict(intensity=self.intensity,
                    background=self.background,
                    backabsorption=self.back_absorption,
                    theta_offset=self.theta_offset)
    def scattering_factors(self, material):
        """
        Returns the scattering factors associated with the material given
        the range of wavelengths/energies used in the probe.
        """
    def __len__(self):
        """
        Returns the number of scattering factors that will be returned for
        the probe.
        """
    def oversample(self, n=6, seed=1):
        """
        Generate an over-sampling of Q to avoid aliasing effects.

        Oversampling is needed for thick layers, in which the underlying
        reflectivity oscillates so rapidly in Q that a single measurement
        has contributions from multiple Kissig fringes.

        Sampling will be done using a pseudo-random generator so that
        accidental structure in the function does not contribute to the
        aliasing.  The generator will usually be initialized with a fixed
        *seed* so that the point selection will not change from run to run,
        but a *seed* of None will choose a different set of points each time
        oversample is called.

        The value *n* is the number of points that should contribute to
        each Q value when computing the resolution.   These will be
        distributed about the nominal measurement value, but varying in
        both angle and energy according to the resolution function.  This
        will yield more points near the measurement and fewer farther away.
        The measurement point itself will not be used to avoid accidental
        bias from uniform Q steps.
        """
    def resolution(self, calc_R):
        """
        Apply resolution function associated with the probe.

        Users who wish to create complex resolution functions, e.g.,
        with wavelength following the TOF feather will need to control
        both the sampling and the resolution calculation.  Probe is the
        natural place for this calculation since it controls both of these.
        """
        R = convolve(self.calc_Q, calc_R, self.Q, self.dQ)
        #R = numpy.interp(self.Q, self.calc_Q, calc_R)
        return R

    def beam_parameters(self, R):
        """
        Apply factors such as beam intensity, background, backabsorption,
        and footprint to the data.
        """
        # (condition)*C is C when condition is True or 0 when False,
        # so (condition)*(C-1)+1 is C when condition is True or 1 when False.
        back = (self.Q<0)*(self.back_absorption.value-1)+1
        R = R*back*self.intensity.value + self.background.value
        return R

    def fresnel(self, substrate=None, surround=None):
        """
        Compute the reflectivity for the probe reflecting from a block of
        material with the given substrate.

        Returns F = R(probe.Q), where R is magnitude squared reflectivity.
        """
        # Doesn't use ProbeCache, but this routine is not time critical
        Srho,Smu = (0,0) if substrate is None else substrate.sld(self)
        Vrho,Vmu = (0,0) if surround is None else surround.sld(self)
        I = numpy.ones_like(self.Q)
        calculator = fresnel.Fresnel(rho=Srho*I, mu=Smu*I,
                                     Vrho=Vrho*I, Vmu=Vmu*I)
        return calculator(Q=self.Q,L=self.L*I)

    def plot(self, theory=None, substrate=None, surround=None, view=None):
        """
        Plot theory against data.

        Need substrate/surround for Fresnel reflectivity
        """
        view = view if view is not None else self.view
        if view == 'linear':
            self.plot_linear(theory=theory)
        elif view == 'log':
            self.plot_log(theory=theory)
        elif view == 'fresnel':
            self.plot_fresnel(theory=theory, substrate=substrate,
                              surround=surround)
        elif view == 'Q**4':
            self.plot_Q4(theory=theory)
        else:
            raise TypeError("incorrect reflectivity view '%s'"%self.view)

    def plot_linear(self, theory=None):
        """
        Plot the data associated with probe.
        """
        import pylab
        if hasattr(self, 'R'):
            pylab.errorbar(self.Q, self.R, 
                           yerr=self.dR, xerr=self.dQ, fmt='.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R, hold=True)
        pylab.yscale('linear')
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('Reflectivity')
    def plot_log(self, theory=None):
        """
        Plot the data associated with probe.
        """
        import pylab
        if hasattr(self,'R'):
            pylab.errorbar(self.Q, self.R, 
                           yerr=self.dR, xerr=self.dQ, fmt='.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R, hold=True)
        pylab.yscale('log')
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('Reflectivity')
    def plot_fresnel(self, theory=None, substrate=None, surround=None):
        """
        Plot the Fresnel reflectivity associated with the probe.
        """
        import pylab
        if substrate is None and surround is None:
            raise TypeError("Fresnel reflectivity needs substrate or surround")
        F = self.fresnel(substrate=substrate,surround=surround)
        if hasattr(self,'R'):
            pylab.errorbar(self.Q, self.R/F, self.dR/F, self.dQ, '.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R/F, hold=True)
        pylab.xlabel('Q (inv Angstroms)')
        if substrate is None:
            name = "air:%s"%(surround.name)
        elif surround is None or isinstance(surround,Vacuum):
            name = substrate.name
        else:
            name = "%s:%s"%(substrate.name, surround.name)
        pylab.ylabel('R/R(%s)'%(name))
    def plot_Q4(self, theory=None):
        """
        Plot the Q**4 reflectivity associated with the probe.
        """
        import pylab
        Q4 = 1e8*self.Q**4
        #Q4[Q4==0] = 1
        if hasattr(self,'R'):
            pylab.errorbar(self.Q, self.R*Q4, self.dR*Q4, self.dQ, '.')
        if theory is not None:
            Q,R = theory
            pylab.plot(Q, R*Q4, hold=True)
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('R (100 Q)^4')
class XrayProbe(Probe):
    """
    X-Ray probe.

    Contains information about the kind of probe used to investigate
    the sample.

    X-ray data is traditionally recorded by angle and energy, rather
    than angle and wavelength as is used by neutron probes.
    """
    def __init__(self, A=None, dA=0, E=None, dE=0, L=None, dL=0, data=None):
        Probe.__init__(self)
        # Use one of wavelength or energy to define the probe.
        # L = h c / E
        # dL**2 = (-h c / E**2 dE)**2 = (L * dE/E)**2
        if L is None:
            L = xsf.xray_wavelength(E)
            dL = L * dE/E
        else:
            E = xsf.xray_energy(L)
            dE = E * dL/L
        Q,dQ = Qresolution(angle=(A,dA),wavelength=(L,dL))

        # Make sure that angle and energy are vectors
        A,dA,E,dE,L,dL = [numpy.ones_like(Q)*v for v in (A,dA,E,dE,L,dL)]

        # Order by increasing Q
        idx = numpy.argsort(Q)
        self.A,self.dA = A[idx],dA[idx]
        self.E,self.dE = E[idx],dE[idx]
        self.L,self.dL = L[idx],dL[idx]
        self.Qo, self.dQ = Q[idx],dQ[idx]
        if data != None:
            R,dR = data
            self.R,self.dR = R[idx],dR[idx]
            
        # By default the calculated points are the measured points.  Use
        # oversample() for a more accurate resolution calculations.
        self._set_calc(self.A,self.E)

    def oversample(self, n=6, seed=1):
        # doc string is inherited from parent (see below)
        rng = numpy.random.RandomState(seed=seed)
        A = rng.normal(self.A[:,None],self.dA[:,None],size=(len(self.dA),n))
        E = rng.normal(self.E[:,None],self.dE[:,None],size=(len(self.dE),n))
        A = A.flatten()
        E = E.flatten()
        self._set_calc(A,E)
    oversample.__doc__ = Probe.oversample.__doc__

    def _set_calc(self, A, E):
        L = xsf.xray_wavelength(E)
        Q = real_to_Q(angle=A,wavelength=L)

        # The selection vector determines which values of Q will be used
        # when calculating and comparing to the data.  This can be a subset
        # of the complete set of data, for example when the data are
        # oversampled.  The indices of the selection vector should be
        # ordered so that the resulting Q vector is increasing.
        idx = numpy.argsort(Q)
        self.calc_A = A[idx]
        self.calc_E = E[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self._sf_E = numpy.unique(self.calc_E)
        self._sf_idx = numpy.searchsorted(self._sf_E, self.calc_E)

    def scattering_factors(self, material):
        # doc string is inherited from parent (see below)
        coh, absorp = xsf.xray_sld(material, energy = self._sf_E, density=1)
        return coh[self._sf_idx], absorp[self._sf_idx], [0]
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def __len__(self):
        return len(self.calc_A)
class NeutronProbe(Probe):
    def __init__(self, A=None, dA=0, L=None, dL=0, data = None):
        Probe.__init__(self)
        Q, dQ = Qresolution(angle=(A,dA),wavelength=(L,dL))

        # Make sure that we are dealing with vectors
        A,dA,L,dL = [numpy.ones_like(Q)*v for v in (A,dA,L,dL)]

        # Probe stores sorted values for convenience of resolution calculator
        idx = numpy.argsort(Q)
        self.A, self.dA = A[idx],dA[idx]
        self.L, self.dL = L[idx],dL[idx]
        self.Qo, self.dQ = Q[idx],dQ[idx]
        if data != None:
            R,dR = data
            self.R,self.dR = R[idx],dR[idx]

        # By default the calculated points are the measured points.  Use
        # oversample() for a more accurate resolution calculations.
        self._set_calc(self.A,self.L)

    def oversample(self, n=6, seed=1):
        # doc string is inherited from parent (see below)
        rng = numpy.random.RandomState(seed=seed)
        A = rng.normal(self.A[:,None],self.dA[:,None],size=(len(self.dA),n))
        L = rng.normal(self.L[:,None],self.dL[:,None],size=(len(self.dL),n))
        A = A.flatten()
        L = L.flatten()
        self._set_calc(A,L)
    oversample.__doc__ = Probe.oversample.__doc__

    def _set_calc(self, A, L):
        Q = real_to_Q(angle=A, wavelength=L)

        idx = numpy.argsort(Q)
        self.calc_A = A[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self._sf_L = numpy.unique(self.calc_L)
        self._sf_idx = numpy.searchsorted(self._sf_L, L)

    def scattering_factors(self, formula):
        # doc string is inherited from parent (see below)
        coh, absorp, incoh = nsf.neutron_sld(formula,
                                             wavelength=self._sf_L,
                                             density=1)
        return coh, absorp[self._sf_idx], incoh
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def __len__(self):
        return len(self.calc_Q)


class PolarizedNeutronProbe(Probe):
    """
    Polarized beam
    
    *xs* (4 x NeutronProbe) is a sequence pp, pm, mp and mm.
    *Aguide* (degrees) is the angle of the guide field
    """
    polarized = True
    def __init__(self, xs=None, Aguide=270):
        Probe.__init__(self)
        pp, pm, mp, mm = xs
        
        dApp, dApm, dAmp, dAmm = dA
            
        Q, dQ = [Qresolution(angle=(Axx,dAxx),wavelength=(Lxx,dLxx))
                 for Axx,dAxx,Lxx,dLxx in zip(A,dA,L,dL)]

        # Make sure that we are dealing with vectors
        A,dA,L,dL = [numpy.ones_like(Q)*v for v in (A,dA,L,dL)]

        # Probe stores sorted values for convenience of resolution calculator
        idx = numpy.argsort(Q)
        self.A, self.dA = A[idx],dA[idx]
        self.L, self.dL = L[idx],dL[idx]
        self.Qo, self.dQ = Q[idx],dQ[idx]
        if data != None:
            R,dR = data
            self.R,self.dR = R[idx],dR[idx]

        # By default the calculated points are the measured points.  Use
        # oversample() for a more accurate resolution calculations.
        self._set_calc(self.A,self.L)

    def oversample(self, n=6, seed=1):
        # doc string is inherited from parent (see below)
        rng = numpy.random.RandomState(seed=seed)
        A = rng.normal(self.A[:,None],self.dA[:,None],size=(len(self.dA),n))
        L = rng.normal(self.L[:,None],self.dL[:,None],size=(len(self.dL),n))
        A = A.flatten()
        L = L.flatten()
        self._set_calc(A,L)
    oversample.__doc__ = Probe.oversample.__doc__

    def _set_calc(self, A, L):
        Q = real_to_Q(angle=A, wavelength=L)

        idx = numpy.argsort(Q)
        self.calc_A = A[idx]
        self.calc_L = L[idx]
        self.calc_Qo = Q[idx]

        # Only keep the scattering factors that you need
        self._sf_L = numpy.unique(self.calc_L)
        self._sf_idx = numpy.searchsorted(self._sf_L, L)

    def scattering_factors(self, formula):
        # doc string is inherited from parent (see below)
        coh, absorp, incoh = nsf.neutron_sld(formula,
                                             wavelength=self._sf_L,
                                             density=1)
        return coh, absorp[self._sf_idx], incoh
    scattering_factors.__doc__ = Probe.scattering_factors.__doc__

    def __len__(self):
        return len(self.calc_Q)

    def plot(self, theory=None, substrate=None, surround=None, view=None):
        """
        Plot theory against data.

        Need substrate/surround for Fresnel reflectivity
        
        An 
        """
        import pylab
        view = view if view is not None else self.view
        if view in ('linear', 'log', 'fresnel', 'Q**4'):
            # Plot available cross sections
            ishold = pylab.ishold()
            for xs in self.xs:
                if xs is not None:
                    if view == 'linear':
                        xs.plot_linear(theory=theory)
                    elif view == 'log':
                        xs.plot_log(theory=theory)
                    elif view == 'fresnel':
                        xs.plot_fresnel(theory=theory, substrate=substrate,
                                        surround=surround)
                    elif view == 'Q**4':
                        xs.plot_Q4(theory=theory)
                    pylab.hold(True)
            if not ishold: pylab.hold(False)
        elif view == 'SA':
            self.plot_asymmetry(theory=theory)
        else:
            raise TypeError("incorrect reflectivity view '%s'"%self.view)

    def plot_asymmetry(self, theory):
        pp,pm,mp,mm = self.xs
        if pp is None or mm is None:
            raise TypeError("cannot form spin asymmetry plot with ++ and --")
        
        if hasattr(pp,'R'):
            Q,SA,dSA = spin_asymmetry(pp.Q,pp.R,pp.dR,mm.Q,mm.R,mm.dR) 
            pylab.errorbar(Q, SA, dSA, '.')
        if theory is not None:
            Q,pp,pm,mp,mm = theory
            Q,SA,_ = spin_asymmetry(Q,pp,None,Q,mm,None)
            pylab.plot(Q, SA, hold=True)
        pylab.xlabel('Q (inv Angstroms)')
        pylab.ylabel('spin asymmetry (R+ - R-)/(R+ + R-)')

def spin_asymmetry(Qp,Rp,dRp,Qm,Rm,dRm):
    """
    Compute spin asymmetry for R+,R-.
    
    Returns *Q*, *SA*, *dSA*.

    Spin asymmetry, *SA*, is::
    
        SA = (R+ - R-)/(R+ + R-)
    
    Uncertainty *dSA* follows from propagation of error::
    
        dSA^2 = SA^2 ( (1/(R+ - R-) - 1/(Rp + R-))^2 dR+^2
                        (1/(R+ - R-) + 1/(Rp + R-))^2 dR-^2 )

    The inputs (*Qp*, *Rp*, *dRp*) and (*Qm*, *Rm*, *dRm*) are measurements 
    for the ++ and -- cross sections respectively.  If *dRp*, *dRm* are None,
    then the returned uncertainty will also be None.
    """
    Rm = interp(Qm,Qp,Rp)
    v = (Rp-Rm)/(Rp+Rm)
    if dRp is not None:
        dRm = interp(Qm,Qp,dRp)
        dvsq = v**2 * ( (1/(Rp-Rm) - 1/(Rp+Rm))**2 * dRp
                        + (1/(Rp-Rm) + 1/(Rp+Rm))**2 * dRm )
        return Qp, v, sqrt(dvsq)
    else:
        return Qp, v, None


'''
class XrayData:
    """
    X-ray reflectivity is a triple (angle, energy, data).

    Base parameters are

    A,dA (degrees)
        angle and uncertainty
    E,dE (keV)
        energy and uncertainty
    R,dR (unitless)
        reflectivity data and uncertainty

    From this we can compute

    L,dL (Angstroms)
        wavelength
    Q,dQ (inv Angstrom**2)
        reciprical space (Angstrom ** -2)
    probe (XrayProbe)
        theory function probe for looking up scattering densities

    The resulting data is stored in increasing order by Q.
    """
    def __init__(self, angle=None, energy=None, data=None):
        QdQ,LdL = xray_AE_to_QL(angle=angle,energy=energy)
        self.A,self.dA = angle
        self.E,self.dE = energy
        self.L,self.dL = LdL
        self.Q,self.dQ = QdQ
        self.R,self.dR = data

        # Create a default probe for computing scattering values
        self.probe = XrayProbe(angle=(self.A,self.dA),
                               energy=(self.E,self.dE),
                               data=(self.R,self.dR))


class NeutronData:
    """
    Neutron reflectivity is a triple (angle, wavelength, data).

    Base parameters are

    A,dA (degrees)
        angle and uncertainty
    L,dL (Angstrom)
        wavelength and uncertainty
    R,dR (unitless)
        reflectivity data and uncertainty

    From this we can compute

    Q,dQ (inv Angstrom**2)
        reciprical space (Angstrom ** -2)
    probe (NeutronProbe)
        theory function probe for looking up scattering densities

    The resulting data is stored in increasing order by Q.
    """
    def __init__(self, angle=None, wavelength=None, data=None):
        QdQ,LdL = real_to_Q(angle=angle,wavelength=wavelenth)
        self.A,self.dA = angle
        self.L,self.dL = LdL
        self.Q,self.dQ = QdQ
        self.R,self.dR = data

        # Create a default probe for computing scattering values
        self.probe = NeutronProbe(angle=(self.A,self.dA),
                                  wavelength=(self.L,self.dL),
                                  data=(self.R,self.dR))
'''