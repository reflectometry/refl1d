# This program is public domain
# Author: Paul Kienzle

import numpy
from numpy import inf
from scipy.special import erf

class Microslabs:
    """
    Manage the micro slab representation of a model.

    In order to compute reflectivity, we need a series of slabs with thickness,
    roughness and scattering potential for each slab.  Because scattering
    potentials are probe dependent we store an array of potentials for each
    probe value.

    The space for the slabs is saved even after reset, in preparation for a
    new set of slabs from different fitting parameters.

    Example
    =======

    The following example shows how to fill a slab model from layers and
    use it to compute reflectivity::

        slabs.clear()
        for layer in model:
            w,sigma,rho,mu,rho_M,theta_M = layer.render()
            slabs.extend(w=w, sigma=sigma, rho=rho, mu=mu,
                         rho_M=rho_M, theta_M=theta_M)
        w,sigma = slabs.w,slabs.sigma
        rho,mu = slabs.rho,slabs.mu
        rho_M,theta_M = slabs.rho_M,slabs.theta_M
        R = refl(kz,w,rho=rho,mu=mu,sigma=sigma, rho_M=rho_M, theta_M=theta_M)
        figure(2)
        plot(kz,R,label='reflectivity')
    """
    def __init__(self, nprobe):
        self._num_slabs = 0
        # _slabs contains the 1D objects w, sigma, rho_M, theta_M of len n
        # _slabsQ contains the 2D objects rho, mu
        self._slabs = numpy.empty(shape=(0,4))
        self._slabsQ = numpy.empty(shape=(0,nprobe,2))

    def clear(self):
        """
        Reset the slab model so that none are present.
        """
        self._num_slabs = 0

    def __len__(self):
        return self._num_slabs
    def repeat(self, start=0, count=1):
        """
        Extend the model so that there are *count* versions of the slabs
        from *start* to the final slab.

        For a list L, this would be L.extend(L[start:]*(count-1))
        """
        # For now use the dumb implementation; a better implementation
        # would remember the repeats and pre-calculate the matrix product
        # for the repeating region, saving much work later.  This has
        # to work in conjunction with interfaces and with magnetic signals
        # which vary across the repeated layers, so perhaps it is not so useful.
        repeats = count-1
        end = len(self)
        length = end-start
        fromidx = slice(start,end)
        toidx = slice(end,end+repeats*length)
        self._reserve(repeats*length)
        self._slabs[toidx] = numpy.tile(self._slabs[fromidx],[repeats,1])
        self._slabsQ[toidx] = numpy.tile(self._slabsQ[fromidx],[repeats,1,1])
        self._num_slabs += repeats*length

    def _reserve(self, nadd):
        """
        Reserve space for at least *nadd* slabs.
        """
        ns,nl,_ = self._slabsQ.shape
        if ns < self._num_slabs + nadd:
            new_ns = self._num_slabs + nadd + 50
            self._slabs.resize((new_ns, 4))
            self._slabsQ.resize((new_ns, nl, 2))

    def extend(self, w=0, sigma=0, rho=0, mu=0, rho_M=0, theta_M=0):
        """
        Extend the micro slab model with the given layers.
        """
        nadd = len(w)
        self._reserve(nadd)
        idx = slice(self._num_slabs, self._num_slabs+nadd)
        self._num_slabs += nadd
        self._slabs[idx,0] = w
        self._slabs[idx,1] = sigma
        self._slabs[idx,2] = rho_M
        self._slabs[idx,3] = theta_M
        self._slabsQ[idx,:,0] = rho
        self._slabsQ[idx,:,1] = mu

    def interface(self, I):
        """
        Interfaces act to smear the microslabs after the fact.  This
        allows more flexibility than trying to compute the effects
        of roughness on non-flat layers.
        """
        print "Ignoring special interface on the top of the stack"
        pass

    def _w(self):
        return self._slabs[:self._num_slabs,0]
    def _sigma(self):
        return self._slabs[:self._num_slabs-1,1]
    def _rho(self):
        return self._slabsQ[:self._num_slabs,:,0].T
    def _mu(self):
        return self._slabsQ[:self._num_slabs,:,1].T
    def _rho_M(self):
        return self._slabs[:self._num_slabs,2].T
    def _theta_M(self):
        return self._slabs[:self._num_slabs,3].T
    w = property(_w, doc="Thickness (A)")
    sigma = property(_sigma, doc="1-sigma Gaussian roughness (A)")
    rho = property(_rho, doc="Scattering length density (10^-6 number density)")
    mu = property(_mu, doc="Absorption (10^-6 number density)")
    rho_M = property(_rho_M, doc="Magnetic scattering")
    theta_M = property(_theta_M, doc="Magnetic scattering angle")

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
        mu = numpy.vstack([self.mu[0,:]]*2).T.flatten()
        ws = numpy.cumsum(self.w[1:-1])
        z = numpy.vstack([numpy.hstack([-10,0,ws]),
                          numpy.hstack([0,ws,ws[-1]+10])]).T.flatten()
        return z,rho,mu

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
        z = numpy.arange(left,right,dz)
        roughness = self.limited_sigma(limit=roughness_limit)
        rho = build_profile(z, self.w, roughness, self.rho[0])
        mu = build_profile(z, self.w, roughness, self.mu[0])
        return z,rho,mu


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
    for i,v in enumerate(value):
        zo = z[idx[i]:idx[i+1]]
        if i==0:
            lvalue = 0
            lblend = 0
        else:
            lvalue = value[i-1]
            lblend = blend(zo-offset[i],roughness[i-1])
        if i >= len(value)-1:
            rvalue = 0
            rblend = 0
        else:
            rvalue = value[i+1]
            rblend = blend(offset[i+1]-zo,roughness[i])
        mvalue = value[i]
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
