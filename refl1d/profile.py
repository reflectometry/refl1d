# This program is public domain
# Author: Paul Kienzle
r"""
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

    >>> S = Microslabs(nprobe=1, dz=2)
    >>> S.clear()
    >>> S.append(w=0, rho=2.07)

Next add the interface.  This uses :meth:`microslabs` to select
the points at which the interface is evaluated, much like you
would do when defining your own special layer type.  Note that the
points Pz are in the center of the micro slabs.  The width of the
final slab may be different.  You do not need to use fixed width
microslabs if you can more efficiently represent the profile with
a smaller number of variable width slabs, but :meth:`contract_profile`
serves the same purpose with less work on your part.

    >>> from numpy import tanh
    >>> Pw, Pz = S.microslabs(20)
    >>> print("widths = %s ..."%(" ".join("%g"%v for v in Pw[:5])))
    widths = 2 2 2 2 2 ...
    >>> print("centers = %s ..."%(" ".join("%g"%v for v in Pz[:5])))
    centers = 1 3 5 7 9 ...
    >>> rho = (1-tanh((Pz-10)/5))/2*(2.07-4.5)+4.5
    >>> S.extend(w=Pw, rho=[rho])

Finally, add the incident medium and see the results.  Note that *rho*
is a matrix, with one column for each incident energy.  We are only
using one energy so we only show the first column.

    >>> S.append(w=0, rho=4.5)
    >>> print("width = %s ..."%(" ".join("%g"%v for v in S.w[:5])))
    width = 0 2 2 2 2 ...
    >>> print("rho = %s ..."%(" ".join("%.2f"%v for v in S.rho[0, :5])))
    rho = 2.07 2.13 2.21 2.36 2.63 ...

 Since *irho* and *sigma* were not specified, they will be zero.

    >>> print("sigma = %s ..."%(" ".join("%g"%v for v in S.sigma[:5])))
    sigma = 0 0 0 0 0 ...
    >>> print("irho = %s ..."%(" ".join("%g"%v for v in S.irho[0, :5])))
    irho = 0 0 0 0 0 ...
"""
from __future__ import division, print_function

import numpy as np
from numpy import inf, nan, isnan
from scipy.special import erf

from .reflectivity import BASE_GUIDE_ANGLE as DEFAULT_THETA_M

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
        # rhoM, thetaM will contain magnetic moment and angle
        self._slabs = np.empty(shape=(0, 2))
        self._slabs_rho = np.empty(shape=(0, nprobe, 2))
        self.rhoM = None  # type: np.ndarray
        self.thetaM = None  # type: np.ndarray
        self._slabs_mag = np.empty(shape=(0, nprobe, 2))
        self.dz = dz
        self._magnetic_sections = []
        self._z_left = self._z_right = 0.
        self._z_offset = 0.

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
        edges = np.arange(0, thickness + self.dz, self.dz, dtype='d')
        edges[-1] = thickness
        centers = (edges[1:] + edges[:-1]) / 2
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
        repeats = count - 1
        end = len(self)
        length = end - start
        fromidx = slice(start, end)
        toidx = slice(end, end + repeats * length)
        self._reserve(repeats * length)
        self._slabs[toidx] = np.tile(self._slabs[fromidx], [repeats, 1])
        self._slabs_rho[toidx] = np.tile(self._slabs_rho[fromidx], [repeats, 1, 1])
        self._num_slabs += repeats * length

        # Replace interface on the top
        self._slabs[self._num_slabs - 1, 1] = interface

        if self._magnetic_sections:
            raise NotImplementedError("Repeated magnetic layers not implemented")

    def _reserve(self, nadd):
        """
        Reserve space for at least *nadd* slabs.
        """
        ns, nl, _ = self._slabs_rho.shape
        if ns < self._num_slabs + nadd:
            new_ns = self._num_slabs + nadd + 50
            self._slabs = self._slabs.copy()
            self._slabs.resize((new_ns, 4), refcheck=False)
            self._slabs_rho = self._slabs_rho.copy()
            self._slabs_rho.resize((new_ns, nl, 2), refcheck=False)

    def extend(self, w=0, sigma=0, rho=0, irho=0):
        """
        Extend the micro slab model with the given layers.
        """
        nadd = len(w)
        self._reserve(nadd)
        idx = slice(self._num_slabs, self._num_slabs + nadd)
        self._num_slabs += nadd
        self._slabs[idx, 0] = w
        self._slabs[idx, 1] = sigma
        self._slabs_rho[idx, :, 0] = np.asarray(rho).T
        self._slabs_rho[idx, :, 1] = np.asarray(irho).T

    def append(self, w=0, sigma=0, rho=0, irho=0):
        """
        Extend the micro slab model with a single layer.
        """
        #self.extend(w=[w], sigma=[sigma], rho=[rho], irho=[irho])
        #return
        self._reserve(1)
        self._slabs[self._num_slabs, 0] = w
        self._slabs[self._num_slabs, 1] = sigma
        self._slabs_rho[self._num_slabs, :, 0] = rho
        self._slabs_rho[self._num_slabs, :, 1] = irho
        self._num_slabs += 1

    def add_magnetism(self, anchor, w, rhoM=0, thetaM=DEFAULT_THETA_M, sigma=0):
        """
        Add magnetic layers.

        Note that *sigma* is a pair *(interface_below, interface_above)*
        representing the magnetic roughness, which may be different
        from the nuclear roughness at the layer boundaries.
        """
        w = np.asarray(w, 'd')
        if np.isscalar(sigma):
            sigma = (sigma, sigma)
        self._magnetic_sections.append((np.vstack((w, rhoM, thetaM)),
                                        anchor, sigma))

    def thickness(self):
        """
        Total thickness of the profile.

        Note that thickness includes the thickness of the substrate and
        surface layers.  Normally these will be zero, but the contract
        profile operation may result in large values for either.
        """
        return np.sum(self._slabs[1:self._num_slabs, 0])

    @property
    def w(self):
        "Thickness (A)"
        return self._slabs[:self._num_slabs, 0]

    @property
    def sigma(self):
        "rms roughness (A)"
        return self._slabs[:self._num_slabs - 1, 1]

    @property
    def surface_sigma(self):
        "roughness for the current top layer, or nan if substrate"
        return self._slabs[self._num_slabs - 1, 1] if self._num_slabs > 0 else nan

    @property
    def rho(self):
        "Scattering length density (10^-6 number density)"
        return self._slabs_rho[:self._num_slabs, :, 0].T

    @property
    def irho(self):
        "Absorption (10^-6 number density)"
        return self._slabs_rho[:self._num_slabs, :, 1].T

    @property
    def ismagnetic(self):
        "True if there are magnetic materials in any slab"
        return self._magnetic_sections != []

    def limited_sigma(self, limit=0):
        """
        Limit the roughness by some fraction of layer thickness.

        This function should be called before :meth:`finalize`, but after
        all slabs have been added to the profile.

        *limit* is the number of times sigma has to fit in the layers
        on either side of the interface.  The returned sigma is
        truncated to min(wlow, whigh)/*limit* where wlow is the thickness
        of the layer below the interface, and whigh is the thickness above
        the interface.  A *limit* value of 0 returns the original sigma.
        Although a gaussian inteface extends to infinity, in practice
        setting a *limit* of 3 allows the layer to reach its bulk value,
        with no cross talk between the interfaces.  For very large
        roughnesses, the blending algorithm allows the sld beyond
        the interface to bleed through the entire layer and into the
        next.  In this case the roughness should be the same on both
        sides of the layer to avoid artifacts at the interface.

        Magnetic roughness is ignored for now.
        """
        self.sigma[:] = compute_limited_sigma(self.w, self.sigma, limit)


    def finalize(self, step_interfaces, dA):
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
        """
        if self.ismagnetic:
            self._align_magnetic_and_nuclear()

        self._set_z_range()

        # render step interfaces
        if step_interfaces:
            self._render_interfaces()

        if self.ismagnetic:
            self._contract_magnetic(dA)
        else:
            self._contract_profile(dA)

    def _set_z_range(self):
        """
        Make sure z-range includes 3-sigma around every interface.
        """
        self.w[0] = self.w[-1] = 0
        offset = np.cumsum(self.w[:-1])
        self._z_left = min(-10, np.min(offset - 3*self.sigma))
        self._z_right = max(offset[-1]+10, np.max(offset + 3*self.sigma))

    def _align_magnetic_and_nuclear(self):
        """
        Add magnetic information to the nuclear slabs, introducing new
        slabs as necessary where magnetic and nuclear do not match.
        """
        from .reflmodule import _align_magnetic

        # Nuclear profile (one wavelength only)
        #if self.rho.shape[0] != 1:
        #    raise ValueError("wavelength-dependent magnetism not supported")
        w, sigma, rho, irho = self.w, self.sigma, self.rho[0], self.irho[0]

        # Fill in gaps for magnetic profile
        wM, sigmaM, rhoM, thetaM = self._join_magnetic_sections(gap_size=1e-6)

        # Align nuclear and magnetic
        w, sigma, rho, irho, wM, sigmaM, rhoM, thetaM = [
            np.ascontiguousarray(v, 'd')
            for v in (w, sigma, rho, irho, wM, sigmaM, rhoM, thetaM)
            ]
        output = np.empty((len(w)+len(wM), 6), 'd')
        n = _align_magnetic(w, sigma, rho, irho, wM, sigmaM, rhoM, thetaM, output)

        # Store the resulting profile
        self._reserve(n - self._num_slabs)  # make sure there is space
        self._num_slabs = n
        self.w[:] = output[:n, 0]
        self.sigma[:] = output[:n-1, 1]
        self.rho[0][:] = output[:n, 2]
        self.irho[0][:] = output[:n, 3]
        self.rhoM = output[:n, 4]
        self.thetaM = output[:n, 5]

    def _render_interfaces(self):
        """
        Convert all interfaces into step interfaces by sampling the
        analytic version of the smoothed profile at intervals of dz.

        The interface effects are limited to the surrounding layers.

        Use of contract_profile afterward is strongly recommended, for
        better performance on models with large sections of constant
        scattering potential.
        """
        z = np.arange(self._z_left, self._z_right + 0.5*self.dz, self.dz)
        n_slabs = len(z)
        n_profiles = self.rho.shape[0]
        offsets = np.cumsum(self.w[:-1])  # assumes w[0] == 0 in _set_z_range

        # generate profiles
        rho = np.empty((n_profiles, n_slabs), 'd')
        irho = np.empty((n_profiles, n_slabs), 'd')
        for k in range(n_profiles):
            # Gd support: cycle through wavelength dependent rho/irho
            rho[k] = build_profile(z, offsets, self.sigma, self.rho[k])
            irho[k] = build_profile(z, offsets, self.sigma, self.irho[k])
        if self.ismagnetic:
            rhoM = build_profile(z, offsets, self.sigma, self.rhoM)
            thetaM = build_profile(z, offsets, self.sigma, self.thetaM)

        w = self.dz * np.ones(n_slabs)
        w[0] = w[-1] = 0.

        # update slabs
        self._reserve(n_slabs - self._num_slabs)
        self._num_slabs = n_slabs
        self.w[:] = w
        self.sigma[:] = 0
        self.rho[:,:] = rho
        self.irho[:,:] = irho
        if self.ismagnetic:
            self.rhoM = rhoM
            self.thetaM = thetaM
        self._z_offset = self._z_left

    def _contract_profile(self, dA):
        from .reflmodule import _contract_by_area

        if dA is None:
            return

        # TODO: need a separate implementation for multiple wavelengths
        if self.rho.shape[0] > 1:
            # Note: should at least check for duplicates otherwise thick
            # layers will get extremely slow
            return

        w, sigma, rho, irho = [
            np.ascontiguousarray(v, 'd')
            for v in (self.w, self.sigma, self.rho[0], self.irho[0])
            ]
        #print "final sld before contract", rho[-1]
        n = _contract_by_area(w, sigma, rho, irho, dA)
        self._num_slabs = n
        self.w[:] = w[:n]
        self.rho[0, :] = rho[:n]
        self.irho[0, :] = irho[:n]
        self.sigma[:] = sigma[:n-1]
        #print "final sld after contract", rho[n-1], self.rho[0][n-1], n

    def _contract_magnetic(self, dA):
        from .reflmodule import _contract_mag

        if dA is None:
            return

        # TODO: need a separate implementation for multiple wavelengths
        if self.rho.shape[0] > 1:
            # Note: should at least check for duplicates otherwise thick
            # layers will get extremely slow
            return

        w, sigma, rho, irho, rhoM, thetaM = \
            [np.ascontiguousarray(v, 'd')
             for v in (self.w, self.sigma, self.rho[0], self.irho[0], self.rhoM, self.thetaM)]
        #print "final sld before contract", rho[-1]
        n = _contract_mag(w, sigma, rho, irho, rhoM, thetaM, dA)
        self._num_slabs = n
        self.w[:] = w[:n]
        self.rho[0][:] = rho[:n]
        self.irho[0][:] = irho[:n]
        self.rhoM = rhoM[:n]
        self.thetaM = thetaM[:n]
        self.sigma[:] = sigma[:n-1]
        #self.sigma[:] = 0
        #print "final sld after contract", rho[n-1], self.rho[0][n-1], n

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
        if dA is None or smoothness == 0:
            return
        # TODO: refine this so that it can look forward as well as back
        # TODO: also need to avoid changing explicit sigma=0...
        w, sigma, rho, irho = self.w, self.sigma, self.rho[0], self.irho[0]
        step = (abs(np.diff(rho)) + abs(np.diff(irho)))
        step[:-1] *= w[1:-1]  # compute dA of step; substrate uses w=1
        idx = np.nonzero(sigma == 0)[0]
        fix = step[idx] < 3 * dA
        sigma[idx[fix]] = w[idx[fix]] * smoothness

    def step_profile(self):
        """
        Return a step profile representation of the microslab structure.

        Nevot-Croce interfaces are not represented.
        """
        rho = np.vstack([self.rho[0, :], self.rho[0, :]]).T.flatten()
        irho = np.vstack([self.irho[0, :], self.irho[0, :]]).T.flatten()
        if len(self.w) > 2:
            offsets = np.cumsum(self.w[0:-1])
            z = np.vstack([np.hstack([-10, offsets]),
                           np.hstack([offsets, offsets[-1]+10])]).T.flatten()
        else:
            z = np.array([-10, 0, 0, 10])
        return z+self._z_offset, rho, irho

    def magnetic_step_profile(self):
        """
        Return a step profile representation of the microslab structure.

        Nevot-Croce interfaces are not represented.
        """
        z, rho, irho = self.step_profile()
        rhoM = np.vstack([self.rhoM, self.rhoM]).T.flatten()
        thetaM = np.vstack([self.thetaM, self.thetaM]).T.flatten()
        return z+self._z_offset, rho, irho, rhoM, thetaM

    def smooth_profile(self, dz=0.1):
        """
        Return a smooth profile representation of the microslab structure

        Nevot-Croce roughness is approximately represented, though the
        calculation is incorrect for layers with large roughness compared
        to the thickness.

        The returned profile has uniform step size *dz*.
        """
        z = np.arange(self._z_left, self._z_right + 0.5*dz, dz)
        offsets = np.cumsum(self.w) + self._z_offset
        irho = build_profile(z, offsets, self.sigma, self.irho[0])
        rho = build_profile(z, offsets, self.sigma, self.rho[0])
        return z, rho, irho

    def magnetic_smooth_profile(self, dz=0.1):
        """
        Return a profile representation of the magnetic microslab structure.
        """
        z = np.arange(self._z_left, self._z_right + 0.5*dz, dz)
        offsets = np.cumsum(self.w) + self._z_offset
        irho = build_profile(z, offsets, self.sigma, self.irho[0])
        rho = build_profile(z, offsets, self.sigma, self.rho[0])
        rhoM = build_profile(z, offsets, self.sigma, self.rhoM)
        thetaM = build_profile(z, offsets, self.sigma, self.thetaM)
        return z, rho, irho, rhoM, thetaM

    def _join_magnetic_sections(self, gap_size):
        """
        Convert anchored magnetic sections into coarse magnetic slabs.
        """
        # Find the magnetic blocks
        blocks, offsets, sigmas = zip(*self._magnetic_sections)

        #print "blocks", blocks
        #print "offsets", offsets
        #print "sigmas", sigmas
        # * Splice the blocks together with rhoM=0 in the gaps and
        #   thetaM=(thetaM_below+thetaM_above)/2 in the gaps.
        # * Result is:
        #   slices = [(thickness, rhoM, thetaM), (thickness, rhoM, thetaM), ...]
        # * Initialize slices with the magnetism of the substrate, which will
        #   be [thickness=0, rhoM=0, thetaM=first thetaM] unless substrate
        #   magnetism has been specified
        substrate_magnetism = isnan(sigmas[0][0])
        if substrate_magnetism:
            slices = [[[], [], []]]
        else:
            slices = [[[0], [0], [blocks[0][2, 0]]]]
        interfaces = []
        pos = 0
        for i, B in enumerate(blocks):
            anchor = offsets[i]
            w = anchor - pos
            if w >= gap_size:  # Big gap, so need spacer
                # Target average theta between blocks.
                if i == 0:
                    thetaM = B[2, 0]
                    interfaces.append(0)
                else:
                    thetaM = (B[2, 0] + blocks[i - 1][2, -1]) / 2.
                    interfaces.append(sigmas[i - 1][1])
                slices.append([[w], [0], [thetaM]])
                interfaces.append(sigmas[i][0])
            elif w >= -1e-6:
                # Small gap, so add it to the start of the next block
                B[0, 0] += w
                anchor -= w
                if i == 0:
                    if not substrate_magnetism:
                        interfaces.append(sigmas[0][0])
                else:
                    # Use interface_above between blocks which are connected,
                    # ignoring interface_below.
                    interfaces.append(sigmas[i - 1][1])
            else:
                # negative gap should never happen
                raise ValueError("Overlapping magnetic layers at %d" % i)
            slices.append(B)
            nslabs = len(B[0, :])
            interfaces.extend([0] * (nslabs - 1))
            width = np.sum(B[0, :])
            pos = anchor + width

        # Add the final slice
        w = self.thickness() - pos
        theta = blocks[-1][2, -1]
        slices.append([[w], [0], [theta]])
        interfaces.append(sigmas[-1][1])

        wM, rhoM, thetaM = [np.hstack(v) for v in zip(*slices)]
        sigmaM = np.array(interfaces)
        #print "result", wM, rhoM, thetaM, sigmaM
        return wM, sigmaM, rhoM, thetaM

def compute_limited_sigma(thickness, roughness, limit):
    # Limit roughness to the depths of the surrounding layers.  Roughness
    # of the first and last layers interfaces is limited only by the
    # depth of the first and last layers.  We must check explicitly for
    # a pure substrate system since that has no limits on roughness.
    if limit > 0 and len(thickness) > 2:
        s = np.min((thickness[:-1], thickness[1:]), axis=0) / limit
        s[+0] = thickness[+1] / limit
        s[-1] = thickness[-2] / limit
        roughness = np.where(roughness < s, roughness, s)
    return roughness


def build_profile(z, offset, roughness, value):
    """
    Convert a step profile to a smooth profile.

    *z*          calculation points
    *offset*     offset for each interface
    *roughness*  roughness of each interface
    *value*      target value for each slab
    *max_rough*  limit the roughness to a fraction of the layer thickness
    """
    contrast = np.diff(value)
    result = np.zeros_like(z) + value[0]
    for offset_k, sigma_k, contrast_k in zip(offset, roughness, contrast):
        delta = contrast_k * blend(z, sigma_k, offset_k)
        result += delta
    return result


SQRT1_2 = 1. / np.sqrt(2.0)


def blend(z, sigma, offset):
    """
    blend function

    Given a Gaussian roughness value, compute the portion of the neighboring
    profile you expect to find in the current profile at depth z.
    """
    if sigma <= 0.0:
        return 1.0 * (z >= offset)
    else:
        return 0.5 * erf(SQRT1_2 * (z - offset) / sigma) + 0.5
