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
    >>> print("widths = %s ..."%(" ".join("%g"%v for v in Pw[:5])))
    widths = 2 2 2 2 2 ...
    >>> print("centers = %s ..."%(" ".join("%g"%v for v in Pz[:5])))
    centers = 1 3 5 7 9 ...
    >>> rho = (1-tanh((Pz-10)/5))/2*(2.07-4.5)+4.5
    >>> S.extend(w=Pw, rho=[rho])

Finally, add the incident medium and see the results.  Note that *rho*
is a matrix, with one column for each incident energy.  We are only
using one energy so we only show the first column.

    >>> S.append(w=0,rho=4.5)
    >>> print("width = %s ..."%(" ".join("%g"%v for v in S.w[:5])))
    width = 0 2 2 2 2 ...
    >>> print("rho = %s ..."%(" ".join("%.2f"%v for v in S.rho[0,:5])))
    rho = 2.07 2.13 2.21 2.36 2.63 ...

 Since *irho* and *sigma* were not specified, they will be zero.

    >>> print("sigma = %s ..."%(" ".join("%g"%v for v in S.sigma[:5])))
    sigma = 0 0 0 0 0 ...
    >>> print("irho = %s ..."%(" ".join("%g"%v for v in S.irho[0,:5])))
    irho = 0 0 0 0 0 ...
"""

import numpy
from numpy import inf, nan, isnan
from scipy.special import erf
sqrt2=numpy.sqrt(2.0)

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
    def __init__(self, nprobe, dz=1):
        self._num_slabs = 0
        # _slabs contains the 1D objects w, sigma of len n
        # _slabs_rho contains 2D objects rho, irho, with one for each wavelength
        # _slabs_mag contains 1D objects w, sigma, rho, theta of length nmag
        self._slabs = numpy.empty(shape=(0,2))
        self._slabs_rho = numpy.empty(shape=(0,nprobe,2))
        self.dz = dz
        self._z_offset = 0
        self._magnetic_sections = []

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
        # TODO: force dz onto a common boundary to avoid remeshing
        # in the smooth profile function
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
        self._magnetic_sections = []

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
        self._slabs_rho[toidx] = numpy.tile(self._slabs_rho[fromidx],[repeats,1,1])
        self._num_slabs += repeats*length

        # Replace interface on the top
        self._slabs[self._num_slabs-1,1] = interface

        if self._magnetic_sections:
            raise NotImplementedError("Repeated magnetic layers not implemented")

    def _reserve(self, nadd):
        """
        Reserve space for at least *nadd* slabs.
        """
        ns,nl,_ = self._slabs_rho.shape
        if ns < self._num_slabs + nadd:
            new_ns = self._num_slabs + nadd + 50
            # TODO: what's with the sudden need for refcheck?  Is someone
            # else holding a reference to the array?
            self._slabs.resize((new_ns, 4), refcheck=False)
            self._slabs_rho.resize((new_ns, nl, 2), refcheck=False)

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
        self._slabs_rho[idx,:,0] = numpy.asarray(rho).T
        self._slabs_rho[idx,:,1] = numpy.asarray(irho).T

    def append(self, w=0, sigma=0, rho=0, irho=0):
        """
        Extend the micro slab model with a single layer.
        """
        #self.extend(w=[w],sigma=[sigma],rho=[rho],irho=[irho])
        #return
        self._reserve(1)
        self._slabs[self._num_slabs,0] = w
        self._slabs[self._num_slabs,1] = sigma
        self._slabs_rho[self._num_slabs,:,0] = rho
        self._slabs_rho[self._num_slabs,:,1] = irho
        self._num_slabs += 1

    def add_magnetism(self, anchor, w, rhoM=0, thetaM=270., sigma=0):
        """
        Add magnetic layers.
        """
        w = numpy.asarray(w,'d')
        if numpy.isscalar(sigma): sigma = (sigma,sigma)
        self._magnetic_sections.append((numpy.vstack((w,rhoM,thetaM)),
                                        anchor, sigma))

    def thickness(self):
        """
        Total thickness of the profile.

        Note that thickness includes the thickness of the substrate and
        surface layers.  Normally these will be zero, but the contract
        profile operation may result in large values for either.
        """
        return numpy.sum(self._slabs[1:self._num_slabs,0])

    def interface(self, I):
        """
        Interfaces act to smear the microslabs after the fact.  This
        allows more flexibility than trying to compute the effects
        of roughness on non-flat layers.
        """
        raise NotImplementedError('special interfaces not yet supported')
        # TODO: check that Nevot-Croce works between microslab sections
        self.extend(w = 0, sigma=I, rho = self.rho[-1], irho = self.irho[-1])

    @property
    def w(self):
        "Thickness (A)"
        return self._slabs[:self._num_slabs,0]

    @property
    def sigma(self):
        "rms roughness (A)"
        return self._slabs[:self._num_slabs-1,1]
    @property
    def surface_sigma(self):
        "roughness for the current top layer, or nan if substrate"
        return self._slabs[self._num_slabs-1,1] if self._num_slabs>0 else nan

    @property
    def rho(self):
        "Scattering length density (10^-6 number density)"
        return self._slabs_rho[:self._num_slabs,:,0].T

    @property
    def irho(self):
        "Absorption (10^-6 number density)"
        return self._slabs_rho[:self._num_slabs,:,1].T

    @property
    def ismagnetic(self):
        return self._magnetic_sections != []

    def finalize(self, step_interfaces, dA, roughness_limit):
        """
        Rendering complete.

        Call this method after the microslab model has been constructed,
        so any post-rendering processes can be completed.

        In addition to clearing any width from the substrate and
        the surface surround, this will align magnetic and nuclear slabs,
        convert interfaces to step interfaces if desired, and merge slabs
        with similar scattering potentials to reduce computation time.

        *step_interfaces* is True if interfaces should be rendered using
        slabs.

        *dA* is the tolerance to use when deciding if similar layers can
        be merged.

        *roughness_limit* is the maximum
        """
        self.w[0] = self.w[-1] = 0
        self._limited_sigma(limit=roughness_limit)
        if self.ismagnetic:
            self._render_magnetic()
            self._contract_magnetic(dA)
        elif step_interfaces:
            self._render_interfaces()
            self._contract_profile(dA)
        else:
            self._contract_profile(dA)

    def _render_interfaces(self):
        """
        Convert all interfaces into step interfaces by sampling the
        analytic version of the smoothed profile at intervals of dz.

        The interface effects are limited to the surrounding layers.

        Use of contract_profile afterward is strongly recommended, for
        better performance on models with large sections of constant
        scattering potential.
        """
        z, rho, irho = self._build_smooth_profile(dz=self.dz)
        n = len(z)
        w = self.dz*numpy.ones(n)
        self._reserve(n - self._num_slabs)
        self._num_slabs = n
        # TODO: doesn't handle multiple wavelength
        self.w[:] = w
        self.rho[0][:] = rho
        self.irho[0][:] = irho
        self.sigma[:] = 0
        self._z_offset = z[0]
        #print "z_offset", self._z_offset

    def _contract_magnetic(self, dA):
        from .reflmodule import _contract_mag

        # TODO: do we want to use common boundaries for all lambda?
        # TODO: don't throw away other wavelengths
        if dA is None: return
        w,rho,irho,rhoM,thetaM = \
            [numpy.ascontiguousarray(v,'d')
             for v in (self.w,self.rho[0],self.irho[0],self.rhoM, self.thetaM)]
        #print "final sld before contract",rho[-1]
        n = _contract_mag(w,rho,irho,rhoM,thetaM,dA)
        self._num_slabs = n
        self.w[:] = w[:n]
        self.rho[0][:] = rho[:n]
        self.irho[0][:] = irho[:n]
        self.rhoM = rhoM[:n]
        self.thetaM = thetaM[:n]
        self.sigma[:] = 0
        #print "final sld after contract",rho[n-1],self.rho[0][n-1],n

    def _contract_profile(self, dA):
        from .reflmodule import _contract_by_area

        # TODO: do we want to use common boundaries for all lambda?
        # TODO: don't throw away other wavelengths
        if dA is None: return
        w,sigma,rho,irho = \
            [numpy.ascontiguousarray(v,'d')
             for v in (self.w,self.sigma,self.rho[0],self.irho[0])]
        #print "final sld before contract",rho[-1]
        n = _contract_by_area(w,sigma,rho,irho,dA)
        self._num_slabs = n
        self.w[:] = w[:n]
        self.rho[0,:] = rho[:n]
        self.irho[0,:] = irho[:n]
        self.sigma[:] = sigma[:n-1]
        #print "final sld after contract",rho[n-1],self.rho[0][n-1],n

    def _DEAD_apply_smoothness(self, dA, smoothness=0.3):
        """
        Set a guassian interface for layers which have been coalesced using
        the contract_profile function.

        Note that we guess which interfaces this applies to after the fact
        using criteria similar to those used to coalesce the microslabs
        into layers, and so it may apply to layers which are close in
        scattering length density and have zero sigma, but which were
        distinct in the original model.  The displayed profile will show
        the profile used to calculate the reflectivity, so even though
        this behaviour is different from what the user intended, the
        result will not be misleading.

        In a detailed example of a tethered polymer model, smoothness
        was found to be worse than no smoothness, so this function has
        been removed from the execution stream.
        """
        if dA is None or smoothness == 0: return
        #TODO: refine this so that it can look forward as well as back
        #TODO: also need to avoid changing explicit sigma=0...
        w,sigma,rho,irho = self.w,self.sigma,self.rho[0],self.irho[0]
        step = (abs(numpy.diff(rho)) + abs(numpy.diff(irho)))
        step[:-1] *= w[1:-1]  # compute dA of step; substrate uses w=1
        idx = numpy.nonzero(sigma==0)[0]
        fix = step[idx] < 3*dA
        sigma[idx[fix]] = w[idx[fix]]*smoothness

    def _limited_sigma(self, limit=0):
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
        self.sigma[:] = compute_limited_sigma(self.w, self.sigma, limit)

    def step_profile(self):
        """
        Return a step profile representation of the microslab structure.

        Nevot-Croce interfaces are not represented.
        """
        rho = numpy.vstack([self.rho[0,:]]*2).T.flatten()
        irho = numpy.vstack([self.irho[0,:]]*2).T.flatten()
        if len(self.w) > 2:
            ws = numpy.cumsum(self.w[1:-1])
            z = numpy.vstack([numpy.hstack([-10,0,ws]),
                              numpy.hstack([0,ws,ws[-1]+10])]).T.flatten()
        else:
            z = numpy.array([-10,0,0,10])
        return z+self._z_offset,rho,irho

    def smooth_profile(self, dz=1):
        """
        Return a smooth profile representation of the microslab structure

        Nevot-Croce roughness is approximately represented, though the
        calculation is incorrect for layers with large roughness compared
        to the thickness.

        The returned profile has uniform step size *dz*.
        """
        z,rho,irho = self._build_smooth_profile(dz=dz)
        return z+self._z_offset,rho,irho

    def magnetic_profile(self):
        """
        Return a profile representation of the magnetic microslab structure.
        """
        z,rho,irho = self.step_profile()
        rhoM = numpy.vstack([self.rhoM]*2).T.flatten()
        thetaM = numpy.vstack([self.thetaM]*2).T.flatten()
        return z,rho,irho,rhoM,thetaM

    def _render_magnetic(self):
        """
        Render nuclear and magnetic profiles on common slab boundaries.

        Creates rhoM and thetaM columns.
        """
        # TODO: do we need to worry about magnetism bleeding into the
        # substrate or the surface surround.
        self._render_interfaces()
        z = numpy.cumsum(self.w) + (self._z_offset - self.w[0])

        # Fill in gaps for magnetic profile
        wM, rhoM, thetaM, sigmaM = self._join_magnetic_sections()

        rhoM = build_profile(z, wM, sigmaM, rhoM)
        thetaM = build_profile(z, wM, sigmaM, thetaM)
        #print [len(v) for v in w, rho, irho, rhoM, thetaM]

        self.rhoM = rhoM
        self.thetaM = thetaM

    def _join_magnetic_sections(self):
        """
        Convert anchored magnetic sections into coarse magnetic slabs.
        """
        # Find the magnetic blocks
        blocks, offsets, sigmas = zip(*self._magnetic_sections)

        #print "blocks",blocks
        #print "offsets",offsets
        #print "sigmas",sigmas
        # Splice the blocks together with rhoM_gap=0 and
        # thetaM_gap=(thetaM_below+thetaM_above)/2.
        # slices = [(thickness,rhoM,thetaM), (thickness, rhoM, thetaM), ...]
        # initialize slices with the magnetism of the substrate, which will
        # be [thickness=0, rhoM=0, thetaM=first thetaM] unless substrate
        # magnetism has been specified
        substrate_magnetism = isnan(sigmas[0][0])
        if substrate_magnetism:
            slices = [[[], [], []]]
        else:
            slices = [[[0],[0],[blocks[0][2,0]]]]
        interfaces = []
        sigma = None
        pos = 0
        for i,B in enumerate(blocks):
            anchor = offsets[i]
            w = anchor - pos
            if w >= self.dz: # Big gap, so need spacer
                # Target average theta between blocks.
                if i == 0:
                    thetaM = B[2,0]
                    interfaces.append(0)
                else:
                    thetaM = (B[2,0] + blocks[i-1][2,-1])/2.
                    interfaces.append(sigmas[i-1][1])
                slices.append([[w],[0],[thetaM]])
                interfaces.append(sigmas[i][0])
            elif w >= -1e-6:
                # Small gap, so add it to the start of the next block
                B[0,0] += w
                anchor -= w
                if i == 0:
                    if not substrate_magnetism:
                        interfaces.append(sigmas[0][0])
                else:
                    # Use interface_above between blocks which are connected,
                    # ignoring interface_below.
                    interfaces.append(sigmas[i-1][1])
            else:
                # negative gap should never happen
                raise ValueError("Overlapping magnetic layers at %d"%i)
            slices.append(B)
            nslabs = len(B[0,:])
            interfaces.extend([0]*(nslabs-1))
            width = numpy.sum(B[0,:])
            pos = anchor + width

        # Add the final slice
        w = self.thickness() - pos
        theta = blocks[-1][2,-1]
        slices.append([[w],[0],[theta]])
        interfaces.append(sigmas[-1][1])

        wM,rhoM,thetaM = [numpy.hstack(v) for v in zip(*slices)]
        sigmaM = numpy.array(interfaces)
        #print "result", wM, rhoM, thetaM, sigmaM
        return wM,rhoM,thetaM,sigmaM

    def _build_smooth_profile(self, dz):
        thickness = self.thickness()
        left  = 0 - max(10,self.sigma[0]*3)
        right = thickness + max(10,self.sigma[-1]*3)
        z = numpy.arange(left,right+dz,dz)
        # Only show the first wavelength
        rho  = build_profile(z, self.w, self.sigma, self.rho[0])
        irho = build_profile(z, self.w, self.sigma, self.irho[0])
        return z,rho,irho


def compute_limited_sigma(thickness, roughness, limit):
    # Limit roughness to the depths of the surrounding layers.  Roughness
    # of the first and last layers interfaces is limited only by the
    # depth of the first and last layers.  We must check explicitly for
    # a pure substrate system since that has no limits on roughness.
    if limit > 0 and len(thickness)>2:
        s = numpy.min((thickness[:-1],thickness[1:]),axis=0)/limit
        s[ 0] = thickness[ 1]/limit
        s[-1] = thickness[-2]/limit
        roughness = numpy.where(roughness<s,roughness,s)
    return roughness


def build_mag_profile(z, d, v, blends):
    """
    Convert magnetic segments to a smooth profile.
    """

    # TODO this could be faster since we don't need to blend initially.
    s = 0*z
    v = build_profile(z, d, s, v)


def build_profile(z, thickness, roughness, value):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points
    *thickness*  thickness of the layers (first and last values ignored)
    *roughness*  roughness of the interfaces (one less than d)
    *value*      profile being computed
    *max_rough*  limit the roughness to a fraction of the layer thickness
    """
    # Quickly assign a profile ignoring any roughness
    result = value[numpy.searchsorted(numpy.cumsum(thickness[:-1]), z)]
        
    # Then go back to any rough layers and blend properly.
    if not roughness.any(): return result
    rough_layers = roughness.nonzero()[0] # nonzero returns a tuple of arrays
    
    # Make an iterable over rough layers and layers immediately above them,
    # taking care not to repeat any.
    rough_layers = set((rough_layers).tolist()+(rough_layers+1).tolist())
    
    # Find interface depths
    offset = numpy.hstack( (-inf, 0, numpy.cumsum(thickness[1:-1]), inf) )
        
    # gives the layer boundaries in terms of the index of the z
    idx = numpy.searchsorted(z, offset)
    
    # TODO: The following hack makes sure the final z value is calculated.
    # TODO: Make sure it works even when z is wider than the range of offsets.
    if idx[-1] < len(z):
        idx[-1] = len(z)
        
    for i in rough_layers:
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
            
        mblend = 1 - (lblend+rblend)
        result[idx[i]:idx[i+1]] = value[i]*mblend + lvalue*lblend + rvalue*rblend
        
    return result

def blend(z, rough):
    """
    blend function

    Given a Gaussian roughness value, compute the portion of the neighboring
    profile you expect to find in the current profile at depth z.
    """
    if rough <= 0.0:
        return z <= 0.0 # True/False behave as 1/0 for float * and +
    else:
        return 0.5*( 1.0 - erf( z/( rough*sqrt2 ) ) )
