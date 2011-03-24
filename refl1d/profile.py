# This program is public domain
# Author: Paul Kienzle

"""
Scattering length density profile.

In order to render a reflectometry model, the theory function calculator
renders each layer in the model for each energy in the probe.  For slab
layers this is easy: just accumulate the slabs, with the 1-\ $\sigma$ Gaussian
interface width between the slabs.  For freeform or functional layers,
this is more complicated.  The rendering needs to chop each layer into
microslabs and evaluate the profile at each of these slabs.

Example
-------

This example sets up a model which uses tanh to transition from
silicon to gold in 20 |Ang| with 2 |Ang| steps.

First define the profile, and put in the substrate:

    >>> S = Microslabs(nprobe=1,dz=2)
    >>> S.clear()
    >>> S.append(w=0,rho=2.07)

Next add the interface.  This uses :meth:`microslabs` to select
the points at which the interface is evaluated, much like you
would do when defining your own special layer type.  Note that the
points Pz are in the center of the micro slabs.  The width of the
final slab may be different.  You do not need to use fixed width
microslabs if you can more efficiently represent the profile with
a smaller number of variable width slabs, but :meth:`contract_profile`
serves the same purpose with less work on your part.

    >>> from numpy import tanh
    >>> Pw,Pz = S.microslabs(20)
    >>> print "widths = %s ..."%(" ".join("%g"%v for v in Pw[:5]))
    widths = 2 2 2 2 2 ...
    >>> print "centers = %s ..."%(" ".join("%g"%v for v in Pz[:5]))
    centers = 1 3 5 7 9 ...
    >>> rho = (1-tanh((Pz-10)/5))/2*(2.07-4.5)+4.5
    >>> S.extend(w=Pw, rho=[rho])

Finally, add the incident medium and see the results.  Note that *rho*
is a matrix, with one column for each incident energy.  We are only
using one energy so we only show the first column.

    >>> S.append(w=0,rho=4.5)
    >>> print "width = %s ..."%(" ".join("%g"%v for v in S.w[:5]))
    width = 0 2 2 2 2 ...
    >>> print "rho = %s ..."%(" ".join("%.2f"%v for v in S.rho[0,:5]))
    rho = 2.07 2.13 2.21 2.36 2.63 ...

 Since *irho* and *sigma* were not specified, they will be zero.

    >>> print "sigma = %s ..."%(" ".join("%g"%v for v in S.sigma[:5]))
    sigma = 0 0 0 0 0 ...
    >>> print "irho = %s ..."%(" ".join("%g"%v for v in S.irho[0,:5]))
    irho = 0 0 0 0 0 ...
"""

import numpy
from numpy import inf
from .reflmodule import _contract_by_area, _contract_by_step
from .reflectivity import erf
#from scipy.special import erf

class Microslabs(object):
    """
    Manage the micro slab representation of a model.

    In order to compute reflectivity, we need a series of slabs with thickness,
    roughness and scattering potential for each slab.  Because scattering
    potentials are probe dependent we store an array of potentials for each
    probe value.

    Some slab models use non-uniform layers, and so need the additional
    parameter of dz for the step size within the layer.

    The space for the slabs is saved even after reset, in preparation for a
    new set of slabs from different fitting parameters.

    """
    def __init__(self, nprobe, dz=None):
        self._num_slabs = 0
        # _slabs contains the 1D objects w, sigma, rho_M, theta_M of len n
        # _slabsQ contains the 2D objects rho, irho
        self._slabs = numpy.empty(shape=(0,2))
        self._slabsQ = numpy.empty(shape=(0,nprobe,2))
        self._slabsM = []
        self.dz = dz

    def microslabs(self, thickness=0):
        """
        Return a set of microslabs for a layer of the given *thickness*.

        The step size slabs.dz was defined when the Microslabs
        object was created.

        This is a convenience function.  Layer definitions can choose
        their own slices so long as the step size is approximately
        slabs.dz in the varying region.

        :Parameters:
            *thickness* : float | A
                Layer thickness
        :Returns:
            *widths*: vector | A
                Microslab widths
            *centers*: vector | A
                Microslab centers
        """
        edges = numpy.arange(0,thickness+self.dz,self.dz, dtype='d')
        edges[-1] = thickness
        centers = (edges[1:] + edges[:-1])/2
        widths = edges[1:] - edges[:-1]
        return widths, centers

    def clear(self):
        """
        Reset the slab model so that none are present.
        """
        self._num_slabs = 0

    def __len__(self):
        return self._num_slabs

    def repeat(self, start=0, count=1, interface=0):
        """
        Extend the model so that there are *count* versions of the slabs
        from *start* to the final slab.

        This is equivalent to L.extend(L[start:]*(count-1)) for list L.
        """
        # For now use the dumb implementation; a better implementation
        # would remember the repeats and pre-calculate the matrix product
        # for the repeating region, saving much work later.  This has
        # to work in conjunction with interfaces and with magnetic profiles.
        repeats = count-1
        end = len(self)
        length = end-start
        fromidx = slice(start,end)
        toidx = slice(end,end+repeats*length)
        self._reserve(repeats*length)
        self._slabs[toidx] = numpy.tile(self._slabs[fromidx],[repeats,1])
        self._slabsQ[toidx] = numpy.tile(self._slabsQ[fromidx],[repeats,1,1])
        self._num_slabs += repeats*length

        # Replace interface on the top
        self._slabs[self._num_slabs-1,1] = interface

        # TODO: any magnetic sections within the repeat need to be
        # repeated as well

    def _reserve(self, nadd):
        """
        Reserve space for at least *nadd* slabs.
        """
        ns,nl,_ = self._slabsQ.shape
        if ns < self._num_slabs + nadd:
            new_ns = self._num_slabs + nadd + 50
            self._slabs.resize((new_ns, 4))
            self._slabsQ.resize((new_ns, nl, 2))

    def extend(self, w=0, sigma=0, rho=0, irho=0):
        """
        Extend the micro slab model with the given layers.
        """
        nadd = len(w)
        self._reserve(nadd)
        idx = slice(self._num_slabs, self._num_slabs+nadd)
        self._num_slabs += nadd
        self._slabs[idx,0] = w
        self._slabs[idx,1] = sigma
        self._slabsQ[idx,:,0] = numpy.asarray(rho).T
        self._slabsQ[idx,:,1] = numpy.asarray(irho).T

    def append(self, w=0, sigma=0, rho=0, irho=0):
        """
        Extend the micro slab model with a single layer.
        """
        self._reserve(1)
        self._slabs[self._num_slabs,0] = w
        self._slabs[self._num_slabs,1] = sigma
        self._slabsQ[self._num_slabs,:,0] = rho
        self._slabsQ[self._num_slabs,:,1] = irho
        self._num_slabs += 1



    def magnetic(self, anchor, w, rhoM=0, thetaM=0):
        """
        Add magnetic layers.
        """
        self._slabsM.append(anchor,w,rhoM,thetaM)

    def thickness(self):
        """
        Total thickness of the profile.
        """
        return numpy.sum(self._slabs[:self._num_slabs,0])

    def interface(self, I):
        """
        Interfaces act to smear the microslabs after the fact.  This
        allows more flexibility than trying to compute the effects
        of roughness on non-flat layers.
        """
        raise NotImplementedError('special interfaces not yet supported')
        # TODO: check that Nevot-Croce works between microslab sections
        self.extend(w = 0, sigma=I, rho = self.rho[-1], irho = self.irho[-1])

    def _w(self):
        return self._slabs[:self._num_slabs,0]

    def _sigma(self):
        return self._slabs[:self._num_slabs-1,1]

    def _rho(self):
        return self._slabsQ[:self._num_slabs,:,0].T

    def _irho(self):
        return self._slabsQ[:self._num_slabs,:,1].T

    def _rhoM(self):
        raise NotImplementedError
        #return self._slabs[:self._num_slabs,2].T

    def _thetaM(self):
        raise NotImplementedError
        #return self._slabs[:self._num_slabs,3].T

    w = property(_w, doc="Thickness (A)")
    sigma = property(_sigma, doc="1-sigma Gaussian roughness (A)")
    rho = property(_rho, doc="Scattering length density (10^-6 number density)")
    irho = property(_irho, doc="Absorption (10^-6 number density)")
    rhoM = property(_rhoM, doc="Magnetic scattering")
    thetaM = property(_thetaM, doc="Magnetic scattering angle")

    def contract_profile(self, dA):
        # TODO: do we want to use common boundaries for all lambda?
        w,sigma,rho,irho=[numpy.ascontiguousarray(v,'d')
                          for v in self.w,self.sigma,self.rho[0],self.irho[0]]
        n = _contract_by_area(w,sigma,rho,irho,dA)
        self._num_slabs = n
        self.rho[0][:] = rho[:n]
        self.irho[0][:] = irho[:n]
        self.sigma[:] = sigma[:n-1]
        self.w[:] = w[:n]

    def smooth_interfaces(self, dA):
        #TODO: refine this so that it can look back as well as forward
        #TODO: also need to avoid changing explicit sigma=0...
        w,sigma,rho,irho = self.w,self.sigma,self.rho[0],self.irho[0]
        step = w[:-1] * (abs(numpy.diff(rho)) + abs(numpy.diff(irho)))
        idx = numpy.nonzero(sigma[1:]==0)[0]+1
        fix = step[idx] < 3*dA
        sigma[idx[fix]] = w[idx[fix]]/4

    def freeze(self, step=False):
        """
        Generate a consistent set of slabs, expanding interfaces where
        necessary and reconciling differences between the nuclear and
        the magnetic steps.
        """
        raise NotImplementedError

    def limited_sigma(self, limit=0):
        """
        Return the roughness limited by layer thickness.

        *limit* is the number of times sigma has to fit in the layers
        on either side of the interface.  The returned sigma is
        truncated to min(wlo,whi)/*limit* where wlo is the thickness
        of the layer below the interface, and whi is the  thickness above
        the interface.  A *limit* value of 0 returns the original sigma.

        The interface calculation only smears an interface across one
        layer.  This can lead to artifacts with large roughness, and
        a difference between the reflectivity computed from the smooth
        profile and that computed from the step profile and Nevot-Croce
        analytic roughness.  To remove the artifact and make the values
        consistent, the profile can be computed with roughness limited
        by surrounding layer thickness.  A *limit* value of 3 yields
        calculated reflectivity that is indistinguishable up to Qz
        determined by profile step size dz.  Below this value artifacts
        can occur when roughness is large.
        """
        # Limit roughness to the depths of the surrounding layers.  Roughness
        # of the first and last layers interfaces is limited only by the
        # depth of the first and last layers.  We must check explicitly for
        # a pure substrate system since that has no limits on roughness.
        roughness = self.sigma
        thickness = self.w
        if limit > 0 and len(thickness)>2:
            s = numpy.min((thickness[:-1],thickness[1:]),axis=0)/limit
            s[ 0] = thickness[ 1]/limit
            s[-1] = thickness[-2]/limit
            roughness = numpy.where(roughness<s,roughness,s)
        return roughness

    def step_profile(self):
        """
        Return a step profile representation of the microslab structure.

        Nevot-Croce roughness is not represented.
        """
        rho = numpy.vstack([self.rho[0,:]]*2).T.flatten()
        irho = numpy.vstack([self.irho[0,:]]*2).T.flatten()
        if len(self.w) > 2:
            ws = numpy.cumsum(self.w[1:-1])
            z = numpy.vstack([numpy.hstack([-10,0,ws]),
                              numpy.hstack([0,ws,ws[-1]+10])]).T.flatten()
        else:
            z = numpy.array([-10,0,0,10])
        return z,rho,irho

    def smooth_profile(self, dz=1, roughness_limit=0):
        """
        Return a smooth profile representation of the microslab structure

        Nevot-Croce roughness is approximately represented, though the
        calculation is incorrect for layers with large roughness compared
        to the thickness.

        The returned profile has uniform step size *dz*.

        *roughness_limit* is the minimum number of roughness widths that must
        lie within each profile.
        """
        w = numpy.sum(self.w[1:-1])
        left = -self.sigma[0]*3
        right = w+self.sigma[-1]*3
        z = numpy.arange(left,right+dz,dz)
        roughness = self.limited_sigma(limit=roughness_limit)
        rho = build_profile(z, self.w, roughness, self.rho[0])
        irho = build_profile(z, self.w, roughness, self.irho[0])
        return z,rho,irho

def build_profile(z, thickness, roughness, value):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points
    *thickness*  thickness of the layers (first and last values ignored)
    *roughness*  roughness of the interfaces (one less than d)
    *value*      profile being computed
    *max_rough*  limit the roughness to a fraction of the layer thickness
    """

    # Find interface depths
    offset = numpy.hstack( (-inf, 0, numpy.cumsum(thickness[1:-1]), inf) )

    # gives the layer boundaries in terms of the index of the z
    idx = numpy.searchsorted(z, offset)
    # TODO: The following hack makes sure the final z value is calculated.
    # TODO: Make sure it works even when z is wider than the range of offsets.
    if idx[-1] < len(z):
        idx[-1] = len(z)

    # compute the results
    result = numpy.empty_like(z)
    for i,mvalue in enumerate(value):
        zo = z[idx[i]:idx[i+1]]
        if i==0:
            lsigma = lvalue = lblend = 0
        else:
            lsigma = roughness[i-1]
            lvalue = value[i-1]
            lblend = blend(zo-offset[i],lsigma)
        if i >= len(value)-1:
            rsigma = rvalue = rblend = 0
        else:
            rsigma = roughness[i]
            rvalue = value[i+1]
            rblend = blend(offset[i+1]-zo,rsigma)
        #print "zo",i,zo
        #print "lblend",lsigma,lblend
        #print "rblend",rsigma,rblend
        mblend = 1 - (lblend+rblend)
        result[idx[i]:idx[i+1]] = mvalue*mblend + lvalue*lblend + rvalue*rblend
        #result[idx[i]:idx[i+1]] = rvalue*rblend

    return result

def blend(z, rough):
    """
    blend function

    Given a Gaussian roughness value, compute the portion of the neighboring
    profile you expect to find in the current profile at depth z.
    """
    if rough <= 0.0:
        return numpy.where(numpy.greater(z, 0), 0.0, 1.0)
    else:
        return 0.5*( 1.0 - erf( z/( rough*numpy.sqrt(2.0) ) ) )
