# -*- coding: utf-8 -*-
# This program is public domain
# Authors Paul Kienzle, Richard Sheridan
"""
Layer models for polymer systems.

Analytic Self-consistent Field (SCF) Brush profile [#Zhulina]_ [#Karim]_

Analytical Self-consistent Field (SCF) Mushroom Profile [#Adamuţi-Trache]_

Numerical Self-consistent Field (SCF) End-Tethered Polymer Profile [#Cosgrove]_ [#deVoss]_ [#Sheridan]_


.. [#Zhulina] Zhulina, EB; Borisov, OV; Pryamitsyn, VA; Birshtein, TM (1991)
   "Coil-Globule Type Transitions in Polymers. 1. Collapse of Layers
   of Grafted Polymer Chains", Macromolecules 24, 140-149.

.. [#Karim] Karim, A; Douglas, JF; Horkay, F; Fetters, LJ; Satija, SK (1996)
   "Comparative swelling of gels and polymer brush layers",
   Physica B 221, 331-336. `<http://dx.doi.org/10.1016/0921-4526(95)00946-9>`
   
.. [#Adamuţi-Trache] Adamuţi-Trache, M., McMullen, W. E. & Douglas, J. F. 
    Segmental concentration profiles of end-tethered polymers with excluded-
    volume and surface interactions. J. Chem. Phys. 105, 4798 (1996).
    
.. [#Cosgrove] Cosgrove, T., Heath, T., Van Lent, B., Leermakers, F. A. M., 
    & Scheutjens, J. M. H. M. (1987). Configuration of terminally attached 
    chains at the solid/solvent interface: self-consistent field theory and 
    a Monte Carlo model. Macromolecules, 20(7), 1692–1696. 
    doi:10.1021/ma00173a041
    
.. [#deVos] De Vos, W. M., & Leermakers, F. A. M. (2009). Modeling the 
    structure of a polydisperse polymer brush. Polymer, 50(1), 305–316. 
    doi:10.1016/j.polymer.2008.10.025
    
.. [#Sheridan] Sheridan, R. J., Beers, K. L., et. al (2014). Direct observation
    of "surface theta" conditons. {in prep}
"""
from __future__ import division
__all__ = ["PolymerBrush","PolymerMushroom","EndTetheredPolymer","VolumeProfile","layer_thickness"]
import inspect
import numpy
from numpy import real, imag, exp
from bumps.parameter import Parameter

from .model import Layer
from . import util


class PolymerBrush(Layer):
    r"""
    Polymer brushes in a solvent

    Parameters:

        *thickness* the thickness of the solvent layer
        *interface* the roughness of the solvent surface
        *polymer* the polymer material
        *solvent* the solvent material or vacuum
        *base_vf* volume fraction (%) of the polymer brush at the interface
        *base* the thickness of the brush interface (A)
        *length* the length of the brush above the interface (A)
        *power* the rate of brush thinning
        *sigma* rms brush roughness (A)

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    These parameters combine in the following profile formula:

    .. math::

        V(z) &= \left\{
          \begin{array}{ll}
            V_o                        & \mbox{if } z <= z_o \\
            V_o (1 - ((z-z_o)/L)^2)^p  & \mbox{if } z_o < z < z_o + L \\
            0                          & \mbox{if } z >= z_o + L
          \end{array}
        \right. \\
        V_\sigma(z)
           &= V(z) \star
                 \frac{e^{-\frac{1}{2}(z/\sigma)^2}}{\sqrt{2\pi\sigma^2}} \\
        \rho(z) &= \rho_p V_\sigma(z) + \rho_s (1-V_\sigma(z))

    where $V_\sigma(z)$ is volume fraction convoluted with brush
    roughness $\sigma$ and $\rho(z)$ is the complex scattering
    length density of the profile.
    """
    def __init__(self, thickness=0, interface=0, name="brush",
                 polymer=None, solvent=None, base_vf=None,
                 base=None, length=None, power=None, sigma=None):
        self.thickness = Parameter.default(thickness, name="brush thickness")
        self.interface = Parameter.default(interface, name="brush interface")
        self.base_vf = Parameter.default(base_vf, name="base_vf")
        self.base = Parameter.default(base, name="base")
        self.length = Parameter.default(length, name="length")
        self.power = Parameter.default(power, name="power")
        self.sigma = Parameter.default(sigma, name="sigma")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name
        # Constraints:
        #   base_vf in [0,1]
        #   base,length,sigma,thickness,interface>0
        #   base+length+3*sigma <= thickness
    def parameters(self):
        return {'solvent':self.solvent.parameters(),
                'polymer':self.polymer.parameters(),
                'base_vf':self.base_vf,
                'base':self.base,
                'length':self.length,
                'power':self.power,
                'sigma':self.sigma,
                }
                
    def profile(self, z):
        base_vf, base, length, power, sigma \
            = [p.value for p in (self.base_vf, self.base,
               self.length, self.power, self.sigma)]
        base_vf /= 100. # % to fraction
        L0 = base  # if base < thickness else thickness
        L1 = base+length # if base+length < thickness else thickness-L0
        if length == 0:
            v = numpy.ones_like(z)
        else:
            v = (1 - ((z-L0)/(L1-L0))**2)
        v[z<L0] = 1
        v[z>L1] = 0
        brush_profile = base_vf * v**power
        # TODO: we could use Nevot-Croce rather than smearing the profile
        vf = smear(z, brush_profile, sigma)
        return vf

    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw,Pz = slabs.microslabs(thickness)
        # Skip layer if it falls to zero thickness.  This may lead to
        # problems in the fitter, since R(thickness) is non-differentiable
        # at thickness = 0.  "Clip to boundary" range handling will at
        # least allow this point to be found.
        # TODO: consider using this behaviour on all layer types.
        if len(Pw) == 0: return

        Mr,Mi = self.polymer.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try: M,S = M[0],S[0]  # Temporary hack
        except: pass

        vf = self.profile(Pz)
        Pw,vf = util.merge_ends(Pw, vf, tol=1e-3)
        P = M*vf + S*(1-vf)
        Pr, Pi = real(P), imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

def layer_thickness(z):
    """
    Return the thickness of a layer given the microslab z points.

    The z points are at the centers of the bins.  we can use the recurrence
    that boundary b[k] = z[k-1] + (z[k-1] - b[k-1]) to compute the
    total length of the layer.
    """
    return 2 * (numpy.sum(z[-1::-2]) - numpy.sum(z[-2::-2]))

class VolumeProfile(Layer):
    """
    Generic volume profile function

    Parameters:

        *thickness* the thickness of the solvent layer
        *interface* the roughness of the solvent surface
        *material* the polymer material
        *solvent* the solvent material
        *profile* the profile function, suitably parameterized

    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    These parameters combine in the following profile formula::

        sld = material.sld * profile + solvent.sld * (1 - profile)

    The profile function takes a depth z and returns a density rho.

    For volume profiles, the returned rho should be the volume fraction
    of the material.  For SLD profiles, rho should be complex scattering
    length density of the material.

    Fitting parameters are the available named arguments to the function.
    The first argument must be *z*, which is the array of depths at which
    the profile is to be evaluated.  It is guaranteed to be increasing, with
    step size 2*z[0].

    Initial values for the function parameters can be given using name=value.
    These values can be scalars or fitting parameters.  The function will
    be called with the current parameter values as arguments.  The layer
    thickness can be computed as :func:`layer_thickness`.

    """
    # TODO: test that thickness(z) matches the thickness of the layer
    def __init__(self, thickness=0, interface=0, name="VolumeProfile",
                 material=None, solvent=None, profile=None, **kw):
        if interface != 0: raise NotImplementedError("interface not yet supported")
        if profile is None or material is None or solvent is None:
            raise TypeError("Need polymer, solvent and profile")
        self.name = name
        self.thickness = Parameter.default(thickness, name="solvent thickness")
        self.interface = Parameter.default(interface, name="solvent interface")
        self.profile = profile
        self.solvent = solvent
        self.material = material

        # Query profile function for the list of arguments
        vars = inspect.getargspec(profile)[0]
        #print "vars",vars
        if inspect.ismethod(profile): vars = vars[1:]  # Chop self
        vars = vars[1:]  # Chop z
        #print vars
        unused = [k for k in kw.keys() if k not in vars]
        if len(unused) > 0:
            raise TypeError("Profile got unexpected keyword argument '%s'"%unused[0])
        dups = [k for k in vars
                if k in ('thickness','interface','polymer','solvent','profile')]
        if len(dups) > 0:
            raise TypeError("Profile has conflicting argument '%s'"%dups[0])
        for k in vars: kw.setdefault(k,0)
        for k,v in kw.items():
            setattr(self,k,Parameter.default(v,name=k))

        self._parameters = vars

    def parameters(self):
        P = {'solvent':self.solvent.parameters(),
             'material':self.material.parameters(),
             }
        for k in self._parameters:
            P[k] = getattr(self,k)
        return P

    def render(self, probe, slabs):
        Mr,Mi = self.material.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        #M,S = M[0],S[0]  # Temporary hack
        Pw,Pz = slabs.microslabs(self.thickness.value)
        if len(Pw)== 0: return
        kw = dict((k,getattr(self,k).value) for k in self._parameters)
        #print kw
        phi = self.profile(Pz,**kw)
        try:
            if phi.shape != Pz.shape: raise Exception
        except:
            raise TypeError("profile function '%s' did not return array phi(z)"
                            %self.profile.__name__)
        Pw,phi = util.merge_ends(Pw, phi, tol=1e-3)
        P = M*phi + S*(1-phi)
        slabs.extend(rho = [real(P)], irho = [imag(P)], w = Pw)
        #slabs.interface(self.interface.value)


def smear(z, P, sigma):
    """
    Gaussian smearing

    :Parameters:
        *z* | vector
            equally spaced sample times
        *P* | vector
            sample values
        *sigma* | real
            root-mean-squared convolution width
    :Returns:
        *Ps* | vector
            smeared sample values
    """
    if len(z) < 3: return P
    dz = z[1]-z[0]
    if 3*sigma < dz: return P
    w = int(3*sigma/dz)
    G = exp(-0.5*(numpy.arange(-w,w+1)*(dz/sigma))**2)
    full = numpy.hstack( ([P[0]]*w, P, [P[-1]]*w) )
    return numpy.convolve(full,G/numpy.sum(G),'valid')

class PolymerMushroom(Layer):
    """
        Polymer mushrooms in a solvent (volume profile)

        Parameters:

        *z* the SLD depth array
        *delta* interaction parameter
        *vf* not quite volume fraction but pretty close
        *sigma* convolution roughness (A)
        *base* a small plateau just long enough to capture the roughness 
        
        Using analytical SCF methods for gaussian chains, which are then scaled by 
        the radius of gyration of the equivalent free polymer as an approximation to
        results of renormalization group methods [#Adamuţi-Trache]

    .. [#Adamuţi-Trache] Adamuţi-Trache, M., McMullen, W. E. & Douglas, J. F. 
        Segmental concentration profiles of end-tethered polymers with excluded-
        volume and surface interactions. J. Chem. Phys. 105, 4798 (1996).
    """
    
    def __init__(self, thickness=0, interface=0, name="shroom",
                 polymer=None, solvent=None, sigma=0,
                 vf=0, delta=0):
        self.thickness = Parameter.default(thickness, name="shroom thickness")
        self.interface = Parameter.default(interface, name="shroom interface")
        self.delta = Parameter.default(delta, name="delta")
        self.vf = Parameter.default(vf, name="vf")
        self.sigma = Parameter.default(sigma, name="sigma")
        self.solvent = solvent
        self.polymer = polymer
        self.name = name
    def parameters(self):
        return {'solvent':self.solvent.parameters(),
                'polymer':self.polymer.parameters(),
                'delta':self.delta,
                'vf':self.vf,
                'sigma':self.sigma,
                'thickness':self.thickness,
                'interface':self.interface
                }
                
    def profile(self, z):
        delta, sigma, vf, thickness \
            = [p.value for p in self.delta, self.sigma, self.vf, self.thickness]

        return smear(z, MushroomProfile(z, delta, vf, sigma), sigma)


    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw,Pz = slabs.microslabs(thickness)
        # Skip layer if it falls to zero thickness.  This may lead to
        # problems in the fitter, since R(thickness) is non-differentiable
        # at thickness = 0.  "Clip to boundary" range handling will at
        # least allow this point to be found.
        # TODO: consider using this behaviour on all layer types.
        if len(Pw) == 0: return

        Mr,Mi = self.polymer.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try: M,S = M[0],S[0]  # Temporary hack
        except: pass

        phi = self.profile(Pz)
        Pw,phi = util.merge_ends(Pw, phi, tol=1e-3)
        P = M*phi + S*(1-phi)
        Pr, Pi = numpy.real(P), numpy.imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

from numpy import sqrt, pi, hstack, ones_like
from scipy.special import erfc, erfcx

thresh=1e-10
def MushroomProfile(z, delta=0.1, vf=1.0, sigma=1.0):
    """
        Polymer mushrooms in a solvent (volume profile)

        Parameters:

        *z* the SLD depth array
        *delta* interaction parameter
        *vf* not quite volume fraction but pretty close
        *sigma* convolution roughness (A)
        *base* a small plateau just long enough to capture the roughness 
        
        Using analytical SCF methods for gaussian chains, which are then scaled by 
        the radius of gyration of the equivalent free polymer as an approximation to
        results of renormalization group methods [#Adamuţi-Trache]

    .. [#Adamuţi-Trache] Adamuţi-Trache, M., McMullen, W. E. & Douglas, J. F. 
        Segmental concentration profiles of end-tethered polymers with excluded-
        volume and surface interactions. J. Chem. Phys. 105, 4798 (1996).
    """
    
    thickness=layer_thickness(z)
    
    base=3.0*sigma # tail is erf, capture 95% of the mixing
    Rg = (thickness-base) / 4.0 # profile ends by ~4 RG, so we can tether these
    x = (z[(z-base)>=0.0] - base) / Rg
    '''
    mushroom_profile_math has a divide by zero problem at delta=0.
    Fix it by weighted average of the profile above and below a threshold.
    No visual difference when delta is between +-0.001, and there's no
    floating point error until ~+-1e-14.
    '''

    if abs(delta)>thresh:
        mushroom_profile = mushroom_profile_math(x,delta,vf)
    else: # we should RARELY get here
        scale=(delta+thresh)/2.0/thresh             
        mushroom_profile=scale*mushroom_profile_math(x,thresh,vf)        
        mushroom_profile=mushroom_profile+(1.0-scale)*mushroom_profile_math(x,-thresh,vf)  

    try:
        base_profile=ones_like(z[(z-base)<0.0])*mushroom_profile[0] # make the base connect with the profile
    except IndexError:
        base_profile=ones_like(z)*mushroom_profile[0]
        
    return hstack((base_profile,mushroom_profile)) # because appending arrays is hard
    
def mushroom_profile_math(x,delta=.1,vf=.1):
    '''
    new method, rewrite for numerical stability at high delta
    delta==0 causes divide by zero error!! Compensate elsewhere.
    http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
    '''
    
    return ((erfc(x/2.0)- \
        erfcx(2.0*(delta+x/4.0))*exp(-((x/2.0)**2))- \
        erfc(x)+ \
        ((1.0-4.0*delta*(x+2.0*delta))*erfcx(2.0*(delta+x/2.0))+ \
        4.0*delta/sqrt(pi))*exp(-(x**2)) \
        )*vf/(2.0 * delta * erfcx(2.0*delta)))

class EndTetheredPolymer(Layer):
    """
        Polymer end-tethered to an interface in a solvent

        Previous layer should not have roughness! use a spline to simulate it.
        
        Parameters:

        *z* the SLD depth array
        *chi* solvent interaction parameter
        *chi_s* surface interaction parameter
        *h_dry* thickness of the neat polymer layer
        *l_lat* real length per lattice site
        *mn* Number average molecular weight
        *m_seg* real mass per lattice segment
        *pdi* Dispersity (Polydispersity index)
        *thickness* Slab thickness should be greater than the contour length
                of the polymer
        *interface* should be zero
    
    The materials can either use the scattering length density directly,
    such as PDMS = SLD(0.063, 0.00006) or they can use chemical composition
    and material density such as PDMS=Material("C2H6OSi",density=0.965).

    """
    
    def __init__(self, thickness=0, interface=0, name="EndTetheredPolymer",
                 polymer=None, solvent=None, chi=0, chi_s=0, h_dry=None, 
                 l_lat=1, mn=None, m_seg=1, pdi=1):
        if interface != 0: raise NotImplementedError("interface not yet supported")
        if polymer is None or solvent is None or h_dry is None or mn is None:
            raise TypeError("Need polymer, solvent and profile")
        
        self.thickness = Parameter.default(thickness, name="SCF thickness")
        self.interface = Parameter.default(interface, name="SCF interface")
        self.chi   = Parameter.default(chi, name="chi")
        self.chi_s = Parameter.default(chi_s, name="surface chi")
        self.h_dry = Parameter.default(h_dry, name="dry thickness")
        self.l_lat = Parameter.default(l_lat, name="lattice layer length")
        self.mn    = Parameter.default(mn, name="Num. Avg. MW")
        self.m_seg = Parameter.default(m_seg, name="lattice segment mass")
        self.pdi   = Parameter.default(pdi, name="Dispersity")
        self.phi_prev = None
        self.solvent = solvent
        self.polymer = polymer
        self.name = name
        
    def parameters(self):
        return {'solvent':self.solvent.parameters(),
                'polymer':self.polymer.parameters(),
                'chi':self.chi,
                'chi_s':self.chi_s,
                'h_dry':self.h_dry,
                'l_lat':self.l_lat,
                'mn':self.mn,
                'm_seg':self.m_seg,
                'pdi':self.pdi,
                'thickness':self.thickness,
                'interface':self.interface
                }
                
    def profile(self, z):
        phi = SCFprofile(z, chi=self.chi.value, chi_s=self.chi_s.value, 
                 h_dry=self.h_dry.value,l_lat=self.l_lat.value,
                 mn=self.mn.value, m_seg=self.m_seg.value, pdi=self.pdi.value,
                 phi0=self.phi_prev)
        self.phi_prev = phi
        return phi


    def render(self, probe, slabs):
        thickness = self.thickness.value
        Pw,Pz = slabs.microslabs(thickness)
        # Skip layer if it falls to zero thickness.  This may lead to
        # problems in the fitter, since R(thickness) is non-differentiable
        # at thickness = 0.  "Clip to boundary" range handling will at
        # least allow this point to be found.
        # TODO: consider using this behaviour on all layer types.
        if len(Pw) == 0: return

        Mr,Mi = self.polymer.sld(probe)
        Sr,Si = self.solvent.sld(probe)
        M = Mr + 1j*Mi
        S = Sr + 1j*Si
        try: M,S = M[0],S[0]  # Temporary hack
        except: pass

        phi = self.profile(Pz)
        Pw,phi = util.merge_ends(Pw, phi, tol=1e-3)
        P = M*phi + S*(1-phi)
        Pr, Pi = numpy.real(P), numpy.imag(P)
        slabs.extend(rho=[Pr], irho=[Pi], w=Pw)

import numpy as np
from numpy import absolute as abs
import time
from numpy.linalg import norm
from scipy.optimize import  root
from scipy.interpolate import pchip_interpolate as pchip

MINLAT=35

def SCFprofile(z, chi=None, chi_s=None, h_dry=None, l_lat=1, mn=None, 
               m_seg=1, pdi=1, phi0=None, disp=None):
    ''' Generate volume fraction profile from SCFsolve based on real parameters.
    
    More doctext to come.
    '''
    
    # calculate lattice space parameters    
    theta=h_dry/l_lat
    r=int(mn/m_seg-.5)
    
    if phi0 is not None:
        # squeeze or stretch our initial guess to match the input space
        phi0_len=len(phi0)
        z_len=len(z)
        if phi0_len<z_len:
            phi0=np.concatenate((phi0,np.zeros(z_len-phi0_len)))
        elif phi0_len>z_len:
            phi0=phi0[0:z_len]
            
        # determine number of lattice layers needed
        keep = phi0/theta > 1e-6
        layers = sum(keep)
        # convert initial guess to lattice space
        z_lat = np.linspace(l_lat/2,z[keep].max(),num=layers) 
        phi0 = abs(pchip(z[keep],phi0[keep],z_lat))

    # solve the self consistent field equations    
    phi_lat=SCFsolve(chi=chi,chi_s=chi_s,pdi=pdi,
                     theta=theta,r=r,disp=disp,phi0=phi0)
    if disp: print "lattice segments: ", r
    
    # re-dimensionalize the solution
    layers=len(phi_lat)
    keep=z<l_lat*layers
    z_lat=np.linspace(l_lat/2,(layers-.5)*l_lat,num=layers)
    phi=pchip(z_lat,phi_lat,z[keep])
    
    # fill in the end with zeros
    zextra=z[np.logical_not(keep)]
    return np.concatenate((phi,np.zeros_like(zextra)))
    
def SCFsolve(chi=0,chi_s=0,pdi=1,theta=None,r=None,disp=0,phi0=None):
    ''' Solve SCF equations using an initial guess and lattice parameters
    
    More doctext to come.
    '''
    
    sigmainput=theta/r

    default_layers=max(MINLAT,theta/np.sqrt(sigmainput))
    default_phi0=np.linspace(np.sqrt(sigmainput),0,num=default_layers)
    default_phi0=default_phi0.reshape(-1,order='F')
    
    if phi0 is None:
        phi0=default_phi0
        
    if sigmainput>=1:
        raise ValueError('Chains that short cannot be squeezed that high')
    
    if not all([0.0<=x<1.0 for x in phi0]):
        print 'phi0=',phi0
        print [0.0<=x<1.0 for x in phi0]
        raise ValueError('all phi0 values must be between zero and one')
        
    if pdi==1:
        SCFeqns=lambda phi: SCFeqns_Cosgrove(phi,chi,chi_s,theta,pdi,r,disp,p_i)
        p_i=False
    elif pdi>1:
        try:
            p_i=SZdist(pdi,r)
            SCFeqns=lambda phi: SCFeqns_deVos(phi,chi,chi_s,theta,pdi,r,disp,p_i)
        except SZerror:
            print 'PDI below threshold of numerical stability. Defaulting to calculations of uniform polymer.'
            SCFeqns=lambda phi: SCFeqns_Cosgrove(phi,chi,chi_s,theta,pdi,r,disp,p_i)
    else:
        raise ValueError('Invalid PDI')
    
    starttime=time.time()   
    
    # Check if default guess is a better guess than input
    if phi0 is not default_phi0:
        eps=SCFeqns(phi0)
        default_eps=SCFeqns(default_phi0)
        if norm(eps)/norm(default_eps)>1:
            phi0=default_phi0
    
    done=False
    tol=2e-6*theta
    ratio=1.2
    layers=len(phi0)
    growing=False
    shrinking=False
    
    def callback(x,*args,**kwargs):
        _proto_callback(x,disp,theta,layers,tol,ratio)
            
    
    while not done:
        if disp: print "Solving full SCF equation set"
        try:
            result=root(SCFeqns,phi0,method='Krylov',callback=callback,
                    options={'disp':bool(disp),'jac_options':{'method':'gmres'}})
            phi=abs(result.x)
            if disp:
                print 'Solver exit code:',result.status,result.message
                print 'result.success', result.success
                print 'lattice size', layers
        except ShortCircuitError as e:
            phi=e.x
            if disp: print e
            
        if disp: print 'phi(L)/sum(phi) =',phi[-1] / theta * 1e6,'(ppm)'
            
        if phi[-1] > tol:
            if not shrinking:
                newlayers=max(1,round(layers*(ratio-1)))
                if disp: print 'Growing undersized lattice by ', newlayers
                phi0=np.append(phi,np.linspace(phi[-1],0,num=newlayers))
                growing=1
        elif layers>MINLAT and phi[round(layers/ratio)] < tol:
            if growing:
                done=True
            else:
                if disp: print 'Shrinking undersized lattice...'
                phi0=phi[0:round(layers/ratio)]
                shrinking=1
        else:
            done=True
            phi0=phi
            
        layers=len(phi0)
        def callback(x,*args,**kwargs):
            _proto_callback(x,disp,theta,layers,tol,ratio) 
            
    if disp:
        print "execution time:", round(time.time()-starttime,3), "s"
        print "lattice size:", layers
    
    return phi
        
def interpfield(phired,fullsize):
    return pchip(np.arange(len(phired)),phired,
                np.linspace(0,len(phired)-1,num=fullsize))
    
class ShortCircuitError(Exception):
    ''' Special Error to stop root() before a solution is found.
    
    '''
    def __init__(self, value,x):
         self.value = value
         self.x=x
    def __str__(self):
         return repr(self.value)
         
def _proto_callback(x,disp,theta,layers,tol,ratio):
    ''' Special callback to stop root() before solution is found.
    
    '''
    if disp: print "Iterating..."
    if x[-1] > 4*tol:
        raise ShortCircuitError('Stopping, lattice too small',x)
    elif layers>MINLAT and x[min(layers-1,round(layers/ratio))] < 4*tol:
        raise ShortCircuitError('Stopping, lattice too big',x)

from scipy.special import gammaln
# from reflmodule import _calc_g_zs_inner #TODO: not included yet

lambda_1=np.float64(1.0)/6.0 #always assume cubic lattice for now
lambda_0=1.0-2.0*lambda_1
lambda_array=np.array([lambda_1,lambda_0,lambda_1])
def _fzeros(*args):
    return np.zeros(args,dtype=np.float64,order='F')

def SCFeqns_deVos(phi_z,chi,chi_s,theta,pdi,navgsegments,verbose,p_i,fulloutput=False):
    ''' System of SCF equation for uniform terminally attached polymers.
    
        Formatted for input to a nonlinar minimizer or solver.
    '''
    
    # let the solver go negative if it wants, or outside the limits 0..1
    phi_z = abs(phi_z.reshape(-1,order='F'))
    if np.any(phi_z>=1):
        return np.empty_like(phi_z).fill(np.nan)
    phi_z_0 = 1.0 - phi_z
    
    layers=phi_z.size
    cutoff=p_i.size
    sigma=theta/navgsegments
    
    if pdi <= 1.0:
        raise ValueError('de Vos nSCF equations only valid for disperse polymer')
    
    # calculate all needed quantities for new g_z
    delta=_fzeros(layers)
    delta[0]=1.0
    phi_z_avg=calc_phi_z_avg(phi_z)
    phi_z_0_avg=calc_phi_z_avg(phi_z_0)
    
    # calculate new g_z and difference given all inputs
    g_z=phi_z_0*np.exp(chi*(phi_z_avg-phi_z_0_avg)+chi_s*delta)
    u=-np.log(g_z)
    uavg=np.mean(u)
    g_z_norm=g_z*np.exp(uavg)
    
    # calculate needed quantites for new phi_z
    g_zs_ta_norm = calc_g_zs(g_z_norm,1,layers,cutoff)
    c_i_norm=sigma*p_i/np.sum(g_zs_ta_norm,axis=0)
    g_zs_free_ngts_norm = calc_g_zs(g_z_norm,c_i_norm,layers,cutoff)
    
    phi_z_new = calc_phi_z(g_zs_ta_norm,g_zs_free_ngts_norm,g_z_norm)
    eps_z=phi_z-phi_z_new
    
    return eps_z
    
def SCFeqns_Cosgrove(phi_z,chi,chi_s,theta,pdi,segments,verbose,p_i,fulloutput=False):
    ''' System of SCF equation for uniform terminally attached polymers.
    
        Formatted for input to a nonlinar minimizer or solver.
    '''
    
    # let the solver go negative if it wants, or outside the limits 0..1
    phi_z = abs(phi_z.reshape(-1,order='F')) 
    phi_z_0 = 1.0 - phi_z
    layers=phi_z.size
    
    if pdi != 1.0 and verbose:
        print "Cosgrove nSCF assume PDI == 1"
        
    # calculate all needed quantities for new g_z
    delta=_fzeros(layers)
    delta[0]=1.0
    phi_z_avg=calc_phi_z_avg(phi_z)
    phi_z_0_avg=calc_phi_z_avg(phi_z_0)
    
    # calculate new g_z and difference given all inputs
    g_z=phi_z_0*np.exp(chi*(phi_z_avg-phi_z_0_avg)+chi_s*delta)
    u=-np.log(g_z)
    uavg=np.mean(u)
    g_z_norm=g_z*np.exp(uavg)
    
    # calculate needed quantites for new phi_z
    g_ta_norm = calc_g_zs(g_z_norm,1,layers,segments)
    sigma=theta/segments
    p_i=_fzeros(1,segments)
    p_i[0,-1]=1
    c_i_norm=sigma*p_i/np.sum(g_ta_norm[:,segments-1])
    g_free_norm = calc_g_zs(g_z_norm,c_i_norm,layers,segments)
    
    phi_z_new = calc_phi_z(g_ta_norm,g_free_norm,g_z_norm)
    eps_z=phi_z-phi_z_new
    
    return eps_z

def calc_phi_z_avg(phi_z):
    return np.convolve(phi_z,lambda_array,'same')

def calc_g_zs(g_z,c_i,layers,segments):
    
    # initialize     

    g_zs=np.empty((layers,segments),dtype=np.float64,order='F')
    
    # choose special case 
    
    if np.size(c_i)==1:
        # terminally attached chains
        c_i=_fzeros(1,segments)
        g_zs[:,0]=_fzeros(layers)
        g_zs[0,0]=g_z[0]
    else:
        # free chains
        g_zs[:,0]=c_i[0,segments-1]*g_z
    
    # inner loops
    
    # FASTEST: call some custom C code identical to "SLOW" loop
        # beware, this changes g_zs _in_place!_
    # _calc_g_zs_inner(g_z,c_i,g_zs,lambda_0,lambda_1,layers,segments)
    
    # FASTER: use the convolve function to partially vectorize  
    #TODO: using this until _calc_g_zs_inner is included in reflmodule
    pg_zs=g_zs[:,0]    
    for r in range(1,segments):
        pg_zs=g_z*(c_i[0,segments-r-1]+np.convolve(pg_zs,lambda_array,'same'))
        g_zs[:,r]=pg_zs
    
    # SLOW: loop outright, pulling some slicing out of the innermost loop  
#    pg_zs=g_zs[:,0] 
#    for r in range(1,segments):
#        c=c_i[0,segments-r-1]
#        z=0
#        g_zs[z,r]=(pg_zs[z]*lambda_0+pg_zs[z+1]*lambda_1+c)*g_z[z]
#        for z in range(1,(layers-1)):
#            g_zs[z,r]=(pg_zs[z-1]*lambda_1+pg_zs[z]*lambda_0+pg_zs[z+1]*lambda_1+c)*g_z[z]
#        z=layers-1
#        g_zs[z,r]=(pg_zs[z]*lambda_0+pg_zs[z-1]*lambda_1+c)*g_z[z]
#        pg_zs=g_zs[:,r]
               
    return g_zs
    
def calc_phi_z(g_ta,g_free,g_z):
    return np.sum(g_ta*np.fliplr(g_free),axis=1)/g_z
    
def SZdist(pdi,nn):
    ''' Calculate Shultz-Zimm distribution from PDI and number average DP
    
    Shultz-Zimm is a "realistic" distribution for linear polymers. Numerical
    problems arise when the distribution gets too uniform, so we raise an error
    which can be caught to default to an exact uniform calculation.
    '''
    
    if pdi<=1.0:
        raise ValueError('Invalid PDI')
    x=1.0/(pdi-1.0)
    cutoff=9000
    
    ni = np.arange(1,cutoff+1,dtype=np.float64)
    r=ni/nn
    p_ni=np.exp(np.log(x/ni)-gammaln(x+1)+x*r*(np.log(x*r)/r-1))
    
    if any(p_ni>=1):
        raise SZerror('Schultz-Zimm calculation blew up')
    
    mysums=np.cumsum(p_ni)
    keep=np.logical_and(
        np.logical_or(np.array(range(cutoff))<nn , p_ni >= 1.0e-6),
        mysums<1)
        
    return p_ni[keep].reshape(1,-1,order='F')
    
class SZerror(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)