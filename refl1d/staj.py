# This program is in the public domain
# Author: Paul Kienzle
r"""
Read and write staj files

Staj files are the model files for the mlayer and gj2 programs, which are
used as the calculation engine for the reflpak suite. Mlayer supports
unpolarized beam with multilayer models,  and has files ending in
**.staj**. GJ2 supports polarized beam without multilayer models, and
has files ending in **.sta**.
"""

from math import pi

import numpy as np
from bumps.wsolve import wsolve

ERF_FWHM = 2.35482004503095 # 2 * sqrt(2*log(2))
TANH_FWHM = 0.47320111770856327 # 1/2 atanh(erf(1/sqrt(2))) / acosh(sqrt(2))
# Derivation
# ==========
# Converting from error function FWHM to 1-sigma requires solving
# exp(-0.5*(z/1)**2) = 0.5 for z and doubling it, giving 2*sqrt(2*log(2))
#
# Converting from tanh FWHM to 1-sigma error function is more complicated.
# First find C where w is defined as 1-sigma equivalent of tanh, using the
# identity Erf.CDF(z=sigma;w=sigma) = tanh.CDF(z=sigma;w=sigma).
# This simplifies to::
#
#    erf.CDF  = (1+erf(z/(w*sqrt(2)))/2 = (1+erf(1/sqrt(2)))/2
#    tanh.CDF = (1+tanh(C/w*z))/2       = (1+tanh(C))/2
#
#    erf.CDF = tanh.CDF => C = atanh(erf(1/sqrt(2)))
#
# Next find C where w is defined as FWHM, using the equivalent probability
# density function::
#
#    PDF(z) = C/2w * sech(C/w*z)**2
#
# Solving PDF(w/2) = PDF(0)/2 yields::
#
#    Pw = PDF(w/2) = C/2w * sech(C/2)**2
#    Po = PDF(0) = C/2w * sech(0)**2/2 = C/2w
#
#    Pw = Po/2 => sech(C/2)**2 = 1/2
#              => C = 2 acosh(sqrt(2))
#
# To find 1-sigma width given tanh FWHM of w, use the scale factor
# s = C_1_sigma/C_fwhm = 1/2 atanh(erf(1/sqrt(2)))/acosh(sqrt(2))


#----------------------------------------------------------------------------
#The format of a Non-Magnetic Staj file is shown below.  It is an ASCII text
#file composed of lines as follows:
#
#Note that nLayers = nTLayers + nMLayers + nBLayers + 3
#This is the number of top, middle, and bottom layers plus extra profile lines.
#
#Note that nRepeat is the number of repeats of the mLayers profile lines.
#This parameter is ignored by KsRefl.
#
#Line 1:         nTLayers  nMLayers  nBLayers  nRepeat  nFitParams  nRoughSteps
#Line 2:         wavelength  wavelengthDiv  angularDiv  [theta_offset]
#Line 3:         intensity  background  Qmin  Qmax  nQ (data points)
#Line 4:         profileType ('E' for error function, 'H' for tanh)
#Line 5:         datafileName
#Line 6:         <optional outputfileName>
##Sections have an ignored layer followed by the layers for the section
##The ignored layer of the top section contains vacuum SLD
##Layers have rho  mrho  depth  rough  mu
#Line 7 to 7+nL: sections
#Line 7+nL+1:    p1 p2 p3 ... (fitted parameter numbers)
#Line 7+nL+2 to end: constraints
## Constraints often end with a line of garbage characters.
#
#On reading a Staj file the following conversions are made:
#- scattering_length_density = 1e6 rho / (16 pi)
#- magnetic_scat_len_density = 1e6 mrho / (16 pi)
#- roughness_between_layers  = rough
#- absorption                = mu / (2 wavelength)
#
#-------------------------------------------------------------------------------
#The following shows the contents of an sample Non-Magnetic Staj file:
#            1            1            1            1            0           21
#        6.00000       0.160000    0.000300000       0.000000
#        1.00000   1.00000E-010     0.00741408       0.109617            267
#E
#SNS_Ni_Si_3col.txt
#
#  0.000000E+000  0.000000E+000  0.000000E+000  1.000000E-010  0.000000E+000
#  0.000000E+000  0.000000E+000  9.000000E+002  0.000000E+000  0.000000E+000
#  0.000000E+000  0.000000E+000  0.000000E+000  1.000000E-010  0.000000E+000
#  4.452130E-004  0.000000E+000  7.439860E+002  5.308710E+001  0.000000E+000
#  4.452130E-004  0.000000E+000  0.000000E+000  5.308710E+001  0.000000E+000
#  1.279860E-004  0.000000E+000  6.250000E+002  1.814250E+001  0.000000E+000
#
#MQC1=MQC2
#]
#
#-------------------------------------------------------------------------------
#The sample Non-Magnetic Staj File shown above (after conversion of data
#elements) consists of the following layers:
#
#- T0 (Air)  rho =  0.0, depth =   0.0, mu = 0.0, mrho = 0.0
#                                          roughness between layers =  0.0
#- T1 (Air)  rho =  0.0, depth = 900.0, mu = 0.0, mrho = 0.0
#                                          roughness between layers = 22.5
#- M1 (Ni)   rho = 8.86, depth = 744.0, mu = 0.0, mrho = 0.0
#                                          roughness between layers = 7.70
#- B1 (Si)   rho = 2.55, depth = 625.0, mu = 0.0, mrho = 0.0
#
#Note that comments or additional blank lines are not permitted except after
#all required lines (i.e. line 1 through line 6+nLayers).
#
#Note that the number of values on the first line determines if the format is
#for a non-magnetic (6 values) or a magnetic (4 values) Staj file.

class MlayerModel(object):
    r"""
    Model definition used by MLayer program.

    **Attributes:**

    Q values and reflectivity come from a data file with Q, R, dR or
    from simulation with linear spacing from Qmin to Qmax in equal steps:

        *data_file*
            name of the data file, or None if this is simulation only
        *Qmin*, *Qmax*, *num_Q*
            for simulation, Q sample points

    Resolution is defined by wavelength and by incident angle:

        *wavelength*, *wavelength_dispersion*, *angular_divergence*
            resolution is calculated as
            $\Delta Q/Q = \Delta\lambda/\lambda + \Delta\theta/\theta$

    Additional beam parameters correct for intensity, background and
    possibly sample alignment:

        *intensity*, *background*
            incident beam intensity and sample background
        *theta_offset*
            alignment angle correction

    The model is defined in terms of layers, with three sections.  The top
    and bottom section correspond to the fixed layers at the surface and
    the substrate.  The middle section layers can be repeated an arbitrary
    number of times, as defined by the number of repeats attribute.  The
    attributes defining the sections are:

        *num_top* *num_middle* *num_bottom*
            section sizes
        *num_repeats*
            number of times middle section repeats

    Interfaces are split into discrete steps according to a profile,
    either error function or hyperbolic tangent.  For sharp interfaces
    which do not overlap within a layer, the interface is broken into a
    fixed number of slabs with slabs having different widths, but equal
    changes in height.  For broad interfaces, the whole layer is split
    into the same fixed number of slabs, but with each slab having the
    same width.  The following attributes are used:

        *roughness_steps*
            number of roughness steps (13 is coarse; 51 is fine)
        *roughness_profile*
            roughness profile is either 'E' for error function or 'H' for tanh

    Layers have thickness, interface roughness and real and imaginary
    scattering length density (SLD).  Roughness is stored in the file
    using full width at half maximum (FWHM) for the given profile type.
    For convenience, roughness can also be set or queried using a 1-\ $\sigma$
    equivalent roughness on an error function profile.  Regardless,
    layer parameters are represented as vectors with one entry for each
    top, middle and bottom layer using the following attributes:

        *thickness*, *roughness* : float | |Ang|
            layer thickness and FWHM roughness
        *rho*, *irho*, *incoh* : float | |1e-6/Ang^2|
            complex coherent $\rho + j \rho_i$ and incoherent SLD

    Computed attributes are provided for convenience:

        *sigma_roughness* : float | |Ang|
            1-\ $\sigma$ equivalent roughness for erf profile
        *mu*
            absorption cross section (2*wavelength*irho + incoh)

    .. Note::
          The staj files store SLD as $16\pi\rho$, $2\lambda\rho_i$
          with an additional column of 0 for magnetic SLD. This conversion
          happens automatically on read/write. The incoherent cross section
          is assumed to be zero.

    The layers are ordered from surface to substrate.

    Additional attributes are as follows:

        *fitpars*
            individual fit parameter numbers
        *constraints*
            constraints between layers
        *output_file*
            name of the output file

    These can be safely ignored, except perhaps if you want to try to
    compile the constraints into something that can be used by your system.

    **Methods:**

    model = MlayerModel(attribute=value, ...)

        Construct a new MLayer model with the given attributes set.

    model = MlayerModel.load(filename)

        Construct a new MLayer model from a staj file.

    model.set(attribute=value, ...)

        Replace a set of attribute values.

    model.fit_resolution(Q, dQ)

        Choose the best resolution parameters to match the given Q, dQ
        resolution.  Returns the object so that calls can be chained.

    model.resolution(Q)

        Return the resolution at Q for the current resolution parameters.

    model.split_sections()

        Assign top, middle, bottom and repeats to distribute the layers
        across sections.  Returns the object so that calls can be chained.

    model.save(filename)

        Write the model to the given named file.  Raises ValueError if
        the model is invalid.

    **Constructing new files:**

    Staj files can be constructed directly.  The MlayerModel constructor
    can accept all data attributes as key word arguments.  Models require
    at least *data_file*, *wavelength*, *thickness*, *roughness* and *rho*.
    Resolution parameters can be set using model.fit_resolution(Q, dQ).
    Section sizes can be set using model.split_sections().  Everything
    else has reasonable defaults.

    """
    data_file = ""
    Qmin = 0
    Qmax = 0.5
    num_Q = 200
    wavelength = 1
    wavelength_dispersion = 0.01
    angular_divergence = 0.001
    intensity = 1
    background = 0
    theta_offset = 0
    num_top, num_middle, num_bottom = 0, 0, 0
    num_repeats = 1
    roughness_steps = 13
    roughness_profile = 'E'
    thickness = roughness = rho = None
    irho = incoh = 0
    fitpars = []
    constraints = ""
    output_file = ""
    def __init__(self, **kw):
        self.set(**kw)

    def set(self, **kw):
        valid = ('data_file', 'Qmin', 'Qmax', 'num_Q', 'wavelength',
                 'wavelength_dispersion', 'angular_divergence', 'intensity',
                 'background', 'theta_offset',
                 'num_top', 'num_middle', 'num_bottom', 'num_repeats',
                 'roughness_steps', 'roughness_profile', 'sigma_roughness',
                 'thickness', 'roughness', 'rho', 'irho', 'incoh',
                 'fitpars', 'constraints', 'output_file')
        for k, v in kw.items():
            if k not in valid:
                raise TypeError("Unexpected attribute '%s' in Mlayer Model"%k)
            setattr(self, k, v)

    @classmethod
    def load(cls, filename):
        """
        Load a staj file, returning an MlayerModel object
        """
        fin = open(filename, 'r')
        lines = fin.readlines()
        fin.close()

        self = cls()
        self._parse(lines)
        return self

    def save(self, filename):
        """
        Save the staj file
        """
        self._check()
        fid = open(filename, 'w')
        try:
            self._write(fid)
        finally:
            fid.close()

    def FWHMresolution(self, Q):
        r"""
        Return the resolution at Q for mlayer with the current settings
        for wavelength, wavelength divergence and angular divergence.

        Resolution is full-width at half maximum (FWHM), not 1-\ $\sigma$.
        """
        return (abs(Q) * self.wavelength_dispersion
                + 4 * pi * self.angular_divergence) / self.wavelength

    def fit_FWHMresolution(self, Q, dQ, weight=1):
        r"""
        Choose the best dL and dT to match the resolution dQ.

        Given that mlayer uses the following resolution function:

        .. math::

            \Delta Q_k = (|Q_k| \Delta\lambda + 4 \pi \Delta\theta)/\lambda_k

        we can use a linear system solver to find the optimal
        $\Delta \lambda$ and $\Delta \theta$ across our dataset from the
        over-determined system:

        .. math::

          [|Q_k|/\lambda_k, 4\pi/\lambda_k][\Delta\lambda, \Delta\theta]^T
              = \Delta Q_k

        If weights are provided (e.g., $\Delta R_k/R_k$), then weigh each
        point during the fit.

        Given that the experiment is often run with fixed slits at the
        start and end, you may choose to match the resolution across the
        entire $Q$ range, or instead restrict it to just the region where
        the slits are opening.  You will generally want to get the resolution
        correct at the critical edge since that's where it will have the
        largest effect on the fit.

        Returns the object so that operations can be chained.
        """
        A = np.array([abs(Q)/self.wavelength,
                      np.ones_like(Q)*(4*pi/self.wavelength)]).T
        s = wsolve(A, y=dQ, dy=weight)
        self.wavelength_dispersion = s.x[0]
        self.angular_divergence = s.x[1]

        return self

    def split_sections(self):
        """
        Split the given set of layers into sections, putting as many layers
        as possible into the middle section, then the bottom and finally
        the top.

        Returns the object so that operations can be chained.
        """
        self.num_repeats = 1
        n = len(self.thickness)
        if n > 28:
            raise ValueError("A maximum of 28 layers is allowed")
        if n < 2:
            raise ValueError("Must have at least two layers")
        n -= 1  # Incident medium layer
        if n >= 11:
            self.num_middle = 9
        elif n > 2:
            self.num_middle = n-2
        elif n > 1:
            self.num_middle = 1
        else:
            self.num_middle = 0
        n -= self.num_middle
        if n >= 10:
            self.num_bottom = 9
        else:
            self.num_bottom = n-1
        n -= self.num_bottom
        self.num_top = n  # This may be zero if there are less than 3 layers.

        return self

    def __str__(self):
        line = []
        if self.data_file is not "":
            line.append("Data: %s"%self.data_file)
        else:
            line.append("Q: %g to %g in %d steps"
                        %(self.Qmin, self.Qmax, self.num_Q))
        line.append("Wavelength L: %g  dL/L: %g, dTheta: %g"
                    %(self.wavelength, self.wavelength_dispersion,
                      self.angular_divergence))
        line.append("Beam intensity: %g  Background: %g, Theta offset: %g"
                    %(self.intensity, self.background, self.theta_offset))
        profile = 'error function' if self.roughness_profile == 'E' else 'tanh'
        line.append("Interface: %s in %d steps"
                    %(profile, self.roughness_steps))
        w, s = self.thickness, self.roughness
        rho, mu = self.rho, self.mu
        line.append("Layers:")
        line.append(("   " + "%15s "*4)
                    %("Width (A)", "Interface (FWHM)",
                      "Rho (1e-6/A)", "Mu (1e-6/A)"))
        for i in range(len(self.rho)):
            if i == 0:
                name = 'V'
            elif i < self.num_top + 1:
                name = 'T%d'%(i)
            elif i < self.num_top + self.num_middle + 1:
                name = 'M%d'%(i-self.num_top)
            else:
                name = 'B%d'%(i-self.num_top-self.num_middle)
            line.append(("%s:"+("%15g "*4))%(name, w[i], s[i], rho[i], mu[i]))
        if self.constraints != "":
            line.append("Constraints:")
            line.append(self.constraints)
        return "\n".join(line)

    def _check(self):
        """
        Verify that the staj file is correct and ready for writing, filling
        in the details that are missing.
        """
        if ((self.irho is not None and len(self.rho) != len(self.irho))
            or len(self.rho) != len(self.thickness)
            or len(self.rho) != len(self.roughness)):
            raise ValueError("layer parameters have different lengths")

        # Could check if Qmin/Qmax/num_Q matches data, but I don't think
        # it matters, so skip it

        if self.num_top+self.num_middle+self.num_bottom+1 != len(self.rho):
            raise ValueError("section sizes do not match number of layers")

    def _get_mu(self):
        mu = 2*self.wavelength*self.irho + self.incoh
        return mu*np.ones_like(self.rho)
    def _set_mu(self, mu):
        self.irho = mu/(2*self.wavelength)
    mu = property(fget=_get_mu, fset=_set_mu)

    def _get_sigma(self):
        if self.roughness_profile == 'H':
            scale = TANH_FWHM
        else:
            scale = ERF_FWHM
        return self.roughness/scale
    def _set_sigma(self, v):
        if self.roughness_profile == 'H':
            scale = TANH_FWHM
        else:
            scale = ERF_FWHM
        self.roughness = v * scale
    sigma_roughness = property(fget=_get_sigma, fset=_set_sigma)

    def _parse(self, lines):
        #1: num_top num_middle num_bottom num_repeats num_fit rough_steps
        nums = [int(s) for s in lines[0].split()]
        self.num_top, self.num_middle, self.num_bottom, self.num_repeats, \
            _, self.roughness_steps = nums

        #2: wavelength  wavelength_dispersion  angular_divergence [theta_offset]
        nums = [float(s) for s in lines[1].split()]
        self.wavelength, self.wavelength_dispersion, self.angular_divergence \
            = nums[:3]
        self.theta_offset = nums[3] if len(nums) > 3 else 0

        #3: intensity  background  Qmin  Qmax  num_Q
        nums = [float(s) for s in lines[2].split()]
        self.intensity, self.background, self.Qmin, self.Qmax = nums[:4]
        self.num_Q = int(nums[4])

        #4: profile_type ('E' for error function, 'H' for tanh)
        self.roughness_profile = lines[3].strip().upper()[:1]
        if self.roughness_profile not in ('E', 'H'):
            raise ValueError("Expected roughness profile type E or H")

        #5: data_file
        #6: output_file
        self.data_file = lines[4].strip()
        self.output_file = lines[5].strip()

        #nL = num_top+num_middle+num_bottom+3
        #7 to 7+nL: rho mrho depth rough mu
        #ignore the layer before each section
        nL = self.num_top+self.num_middle+self.num_bottom+3
        layers = [[float(v) for v in line.split()]
                  for line in lines[6:6+nL]]
        del layers[self.num_top+1]
        del layers[self.num_top+self.num_middle+1]

        A = np.array(layers)
        self.rho = A[:, 0] * (1e6/16/pi)
        self.irho = A[:, 4] * (1e6/2/self.wavelength)
        self.incoh = A[:, 0] * 0
        self.thickness = A[:, 2]
        self.roughness = A[:, 3]

        #7+nL+1: P1 P2 P3 ...  (fit parameters)
        self.fitpars = [int(s) for s in lines[6+nL].split()]
        #7+nL+2 to end: constraints
        self.constraints = "".join(lines[6+nL+1:])

    def _write(self, fid):
        #1: num_top num_middle num_bottom num_repeats num_fit rough_steps
        fid.write("%d %d %d %d %d %d\n"%(self.num_top, self.num_middle,
                                         self.num_bottom, self.num_repeats,
                                         len(self.fitpars), self.roughness_steps))

        #2: wavelength  wavelength_dispersion  angular_divergence [theta_offset]
        fid.write("%g %g %g %g\n"%(self.wavelength, self.wavelength_dispersion,
                                   self.angular_divergence, self.theta_offset))

        #3: intensity  background  Qmin  Qmax  num_Q
        fid.write("%g %g %g %g %d\n"%(self.intensity, self.background,
                                      self.Qmin, self.Qmax, self.num_Q))

        #4: profile_type ('E' for error function, 'H' for tanh)
        fid.write("%s\n"%self.roughness_profile)

        #5: data_file
        #6: output_file
        fid.write("%s\n%s\n"%(self.data_file, self.output_file))

        #nL = num_top+num_middle+num_bottom+3
        #7 to 7+nL: rho mrho depth rough mu
        #ignore the layer before each section
        rho = self.rho*(16*pi*1e-6)
        mu = self.mu*1e-6
        w, s = self.thickness, self.roughness
        def _write_layer(idx):
            fid.write("%g %g %g %g %g\n"%(rho[idx], 0.,
                                          w[idx], s[idx],
                                          mu[idx]))
        offset = 0
        for n in [self.num_top, self.num_middle, self.num_bottom]:
            _write_layer(offset)
            if n == 0:
                # In the case of only two or three layers, some sections
                # may have no layers, and need to be filled with a repeated
                # value from the next section.
                _write_layer(offset)
            for i in range(n):
                _write_layer(i+offset+1)
            offset += n
        #7+nL+1: P1 P2 P3 ...  (fit parameters)
        fid.write(" ".join(str(p) for p in self.fitpars)+"\n")
        #7+nL+2 to end: constraints
        fid.write(self.constraints)



#==============================================================================
# Translate the Staj file into model representation (magnetic case).
#==============================================================================

#-------------------------------------------------------------------------------
#The format of a Magnetic Staj file is shown below.  It is an ASCII text file
#composed of lines as follows:
#
#Note that nLayers does not count the top layer, just the middle and bottom
#layers (though data for the top (incident) layer is provided) in the file.
#
#Line 1:         wavelength  wavelengthDiv  angularDiv  [aguide]
#Line 2:         intensity  background
#Line 3:         nLayers  nRoughSteps  nFitParam
#Line 4:         Qmin  Qmax  nQ (data points in 'a' datafle)
#Line 5:         Qmin  Qmax  nQ (data points in 'b' datafle)
#Line 6:         Qmin  Qmax  nQ (data points in 'c' datafle)
#Line 7:         Qmin  Qmax  nQ (data points in 'd' datafle)
#Line 8:         profileType ('E' for error function, 'H' for tanh)
#Line 9:         active cross sections (usually 'abcd' or 'ABCD')
#Line 10:        data file (base name without suffix char such as test.refl)
#Line 11:        output file
## Layer information follows in sets of 3 lines x (nL+1).
#Line next (1):  rho   depth   rough  mu
#Line next (2):  mrho  mdepth  mrough (mrho is also known as phi)
#Line next (3):  mtheta
## Fit information fills the remainder of the file
#Line 11+3*(nL+1)+1:  Fit parameters (integers)
#Line 11+3*(nL+1)+2 to end-of-file:  Constraint program
#
#On reading a Staj file the following conversions are made:
#- scattering length density = rho * (1e6 / 16 / pi )
#- complex SLD               = irho * (1e6 / 2 / wavelength)
#- magnetic SLD              = mrho * (1e6 / 16 / pi )
#- structural roughness      = rough
#- magnetic roughness        = mrough
#
#-------------------------------------------------------------------------------
#The following shows the contents of an sample Magnetic Staj file:
#
#        5.00000      0.0500000   1.00000E-005       -90.0000
#        1.00000   1.00000E-010
#            7            7            0
#     0.00613985       0.174845            215
#     0.00613985       0.174845            215
#     0.00613985       0.174845            215
#     0.00613985       0.174845            215
#E
# abcd
#Test.refl
#
#       0.000000       0.000000       0.000000       0.000000
#       0.000000       0.000000       0.000000
#        270.000
#    0.000318683        50.0000        2.00000       0.000000
#       0.000000        50.0000        2.00000
#        270.000
#    0.000329239        80.0000        2.00000       0.000000
#       0.000000        80.0000        2.00000
#        270.000
#    0.000352864        216.000        2.00000       0.000000
#    0.000117621        216.000        2.00000
#        270.000
#    0.000329239        80.0000        2.00000       0.000000
#       0.000000        80.0000        2.00000
#        270.000
#    0.000318683        50.0000        2.00000       0.000000
#       0.000000        50.0000        2.00000
#        270.000
#   9.70124E-005        10.0000        2.00000       0.000000
#       0.000000        10.0000        2.00000
#        270.000
#    0.000104050        100.000        2.00000   1.00000E-009
#       0.000000        100.000        2.00000
#        270.000
#
#-------------------------------------------------------------------------------
#The sample Magnetic Staj File shown above (after conversion of data elements)
#consists of the following layers (with mdepth and mrough not shown):
#
#- MV (Air)   rho =  0.0, depth =   0.0, mu = 0.0, mrho =  0.0, mtheta = 270.0
#                                           roughness between layers =   0.0
#- M1 (Pt)    rho = 6.34, depth =  50.0, mu = 0.0, mrho =  0.0, mtheta = 270.0
#                                           roughness between layers = 0.849
#- M2 (Cu)    rho = 6.55, depth =  80.0, mu = 0.0, mrho =  0.0, mtheta = 270.0
#                                           roughness between layers = 0.849
#- M3 (Co+Ni) rho = 7.02, depth = 216.0, mu = 0.0, mrho = 2.34, mtheta = 270.0
#                                           roughness between layers = 0.849
#- M4 (Cu)    rho = 6.55, depth =  80.0, mu = 0.0, mrho =  0.0, mtheta = 270.0
#                                           roughness between layers = 0.849
#- M5 (Pt)    rho = 6.34, depth =  50.0, mu = 0.0, mrho =  0.0, mtheta = 270.0
#                                           roughness between layers = 0.849
#- M6 (SiO2)  rho = 1.93, depth =  10.0, mu = 0.0, mrho =  0.0, mtheta = 270.0
#                                           roughness between layers = 0.849
#- M7 (Si)    rho = 2.07, depth = 100.0, mu = 0.0, mrho =  0.0, mtheta = 270.0
#
#Note that comments or additional blank lines are not permitted except after
#all required lines (i.e. line 1 through line 11+3*(nLayers+1)).
#
#Note that the number of values on the first line determines if the format is
#for a non-magnetic (6 values) or a magnetic (4 values) Staj file.
#
class MlayerMagnetic(object):
    r"""
    Model definition used by GJ2 program.

    **Attributes:**

    Q values and reflectivity come from a data file with Q, R, dR or
    from simulation with linear spacing from Qmin to Qmax in equal steps:

        *data_file*
            base name of the data file, or None if this is simulation only
        *active_xsec*
            active cross sections (usually 'abcd' for all cross sections)
        *Qmin*, *Qmax*, *num_Q*
            for simulation, Q sample points

    Resolution is defined by wavelength and by incident angle:

        *wavelength*, *wavelength_dispersion*, *angular_divergence*
            resolution is calculated as
            $\Delta Q/Q = \Delta\lambda/\lambda + \Delta\theta/\theta$

    Additional beam parameters correct for intensity, background and
    possibly guide field angle:

        *intensity*, *background*
            incident beam intensity and sample background
        *guide_angle*
            angle of the guide field

    Unlike pure structural models, magnetic models are in one large
    section with no repeats.  The single parameter is the number of
    layers, which is implicit in the length of the layer data and
    does not need to be an explicit attribute.

    Interfaces are split into discrete steps according to a profile,
    either error function or hyperbolic tangent.  For sharp interfaces
    which do not overlap within a layer, the interface is broken into a
    fixed number of slabs with slabs having different widths, but equal
    changes in height.  For broad interfaces, the whole layer is split
    into the same fixed number of slabs, but with each slab having the
    same width. The following attributes are used:

        *roughness_steps*
            number of roughness steps (13 is coarse; 51 is fine)
        *roughness_profile*
            roughness profile is either 'E' for error function or 'H' for tanh

    Layers have thickness, interface roughness and real and imaginary
    scattering length density (SLD).  Roughness is stored in the file
    using full width at half maximum (FWHM) for the given profile type.
    For convenience, roughness can also be set or queried using a 1-\ $\sigma$
    equivalent roughness on an error function profile.  Regardless,
    layer parameters are represented as vectors with one entry for each
    top, middle and bottom layer using the following attributes:

        *thickness*, *roughness* : float | |Ang|
            layer thickness and FWHM roughness
        *rho*, *irho* : float, float | $16 \pi \rho$, $2\lambda\rho_i$
            complex scattering length density
        *mthickness*, *mroughness* : float | |Ang|
            magnetic thickness and roughness
        *mrho* : float | $16 \pi \rho_M$
            magnetic scattering length density
        *mtheta* : float | |deg|
            magnetic angle
        *sigma_roughness*, *sigma_mroughness* : float | |Ang|
            computed 1-\ $\sigma$ equivalent roughness for erf profile

    The conversion from stored $16 \pi \rho$, $2\lambda\rho_i$ to
    in memory $10^6 \rho$, $10^6 \rho_i$  happens automatically on
    read/write.

    The layers are ordered from surface to substrate.

    Additional attributes are as follows:

        *fitpars*
            individual fit parameter numbers

        *constraints*
            constraints between layers

        *output_file*
            name of the output file

    These can be safely ignored, except perhaps if you want to try to
    compile the constraints into something that can be used by your system.

    **Methods:**

    model = MlayerMagnetic(attribute=value, ...)

        Construct a new MLayer model with the given attributes set.

    model = MlayerMagnetic.load(filename)

        Construct a new MLayer model from a sta file.

    model.set(attribute=value, ...)

        Replace a set of attribute values.

    model.fit_resolution(Q, dQ)

        Choose the best resolution parameters to match the given Q, dQ
        resolution.  Returns the object so that calls can be chained.

    model.resolution(Q)

        Return the resolution at Q for the current resolution parameters.

    model.save(filename)

        Write the model to the given named file.  Raises ValueError if
        the model is invalid.

    **Constructing new files:**

    Staj files can be constructed directly.  The MlayerModel constructor
    can accept all data attributes as key word arguments.  Models require
    at least *data_file*, *wavelength*, *thickness*, *roughness* and *rho*.
    Resolution parameters can be set using model.fit_resolution(Q, dQ).
    Everything else has reasonable defaults.

    """
    data_file = ""
    active_xsec = "abcd"
    Qmin = 0
    Qmax = 0.5
    num_Q = 200
    wavelength = 1
    wavelength_dispersion = 0.01
    angular_divergence = 0.001
    intensity = 1
    background = 0
    guide_angle = 270
    roughness_steps = 13
    roughness_profile = 'E'
    thickness = roughness = rho = irho = None
    mthickness = mroughness = mrho = mtheta = None
    fitpars = []
    constraints = ""
    output_file = ""
    def __init__(self, **kw):
        self.set(**kw)

    def set(self, **kw):
        valid = ('data_file', 'active_xsec', 'Qmin', 'Qmax', 'num_Q', 'wavelength',
                 'wavelength_dispersion', 'angular_divergence', 'intensity',
                 'background', 'guide_angle',
                 'roughness_steps', 'roughness_profile',
                 'thickness', 'roughness', 'rho', 'irho', 'sigma_roughness',
                 'mthickness', 'mroughness', 'mrho', 'mtheta', 'sigma_mroughness',
                 'fitpars', 'constraints', 'output_file')
        for k, v in kw.items():
            if k not in valid:
                raise TypeError("Unexpected attribute '%s' in Mlayer Model"%k)
            setattr(self, k, v)

    @classmethod
    def load(cls, filename):
        """
        Load a staj file, returning an MlayerModel object
        """
        fin = open(filename, 'r')
        lines = fin.readlines()
        fin.close()

        self = cls()
        try:
            self._parse(lines)
        except:
            raise ValueError("Improper staj file")
        return self

    def save(self, filename):
        """
        Save the staj file
        """
        self._check()
        fid = open(filename, 'w')
        try:
            self._write(fid)
        finally:
            fid.close()

    def FWHMresolution(self, Q):
        return (abs(Q) * self.wavelength_dispersion
                + 4 * pi * self.angular_divergence) / self.wavelength
    FWHMresolution.__doc__ = MlayerModel.FWHMresolution.__doc__

    def fit_FWHMresolution(self, Q, dQ, weight=1):
        A = np.array([abs(Q)/self.wavelength,
                      np.ones_like(Q)*(4*pi/self.wavelength)])
        s = wsolve(A, y=dQ, dy=weight)
        self.wavelength_dispersion = s.x[0]
        self.angular_divergence = s.x[1]

        return self
    fit_FWHMresolution.__doc__ = MlayerModel.fit_FWHMresolution.__doc__

    def __str__(self):
        line = []
        if self.data_file is not "":
            line.append("Data: %s[%s]"
                        %(self.data_file, self.active_xsec.upper()))
        else:
            line.append("Q: %g to %g in %d steps"
                        %(self.Qmin, self.Qmax, self.num_Q))
        line.append("Wavelength L: %g  dL/L: %g, dTheta: %g"
                    %(self.wavelength, self.wavelength_dispersion,
                      self.angular_divergence))
        line.append("Beam intensity: %g  Background: %g, Guide angle: %g"
                    %(self.intensity, self.background, self.guide_angle))
        profile = 'error function' if self.roughness_profile == 'E' else 'tanh'
        line.append("Interface: %s in %d steps"
                    %(profile, self.roughness_steps))
        w, s = self.thickness, self.roughness
        wm = self.mthickness if self.mthickness is not None else w
        sm = self.mroughness if self.mroughness is not None else s
        rho = self.rho
        irho = self.irho if self.irho is not None else np.zeros_like(w)
        mrho = self.mrho if self.mrho is not None else np.zeros_like(w)
        mtheta = self.mtheta if self.mtheta is not None else 270*np.ones_like(w)
        line.append("Layers:")
        line.append(("    " + ("%15s "*4) + "\n    " + ("%15s "*4))
                    %("Width (A)", "Interface (FWHM)",
                      "Rho (1e-6/A)", "iRho (1e-6/A)",
                      "Mag width", "Mag interface", "Mag rho", "Mag angle (deg)"))
        for i in range(len(self.rho)):
            line.append(("%3d:"+("%15g "*4)+"\n    " + ("%15g "*4))
                        %(i, w[i], s[i], rho[i], irho[i],
                          wm[i], sm[i], mrho[i], mtheta[i]))
        if self.constraints != "":
            line.append("Constraints:")
            line.append(self.constraints)
        return "\n".join(line)

    def _check(self):
        """
        Verify that the staj file is correct and ready for writing, filling
        in the details that are missing.
        """
        ns = [(len(v) if v is not None else 0)
              for v in (self.rho, self.thickness, self.roughness, self.irho,
                        self.mrho, self.mthickness, self.mroughness,
                        self.mtheta)]
        if any((n != ns[0] and n != 0) for n in ns[1:]):
            raise ValueError("layer parameters have different lengths")
        if any((n == 0) for n in ns[:3]):
            raise ValueError("rho, thickness and roughness are required")

        # Could check if Qmin/Qmax/num_Q matches data, but I don't think
        # it matters, so skip it

    def _get_sigma(self):
        if self.roughness_profile == 'H':
            scale = TANH_FWHM
        else:
            scale = ERF_FWHM
        return self.roughness/scale
    def _set_sigma(self, v):
        if self.roughness_profile == 'H':
            scale = TANH_FWHM
        else:
            scale = ERF_FWHM
        self.roughness = v * scale
    sigma_roughness = property(_get_sigma, _set_sigma)

    def _get_msigma(self):
        if self.roughness_profile == 'H':
            scale = TANH_FWHM
        else:
            scale = ERF_FWHM
        return self.mroughness/scale
    def _set_msigma(self, v):
        if self.mroughness_profile == 'H':
            scale = TANH_FWHM
        else:
            scale = ERF_FWHM
        self.mroughness = v * scale
    sigma_mroughness = property(_get_msigma, _set_msigma)

    def _parse(self, lines):
        #1: wavelength  wavelength_dispersion  angular_divergence [aguide]
        nums = [float(s) for s in lines[0].split()]
        self.wavelength, self.wavelength_dispersion, self.angular_divergence \
            = nums[:3]
        self.guide_angle = nums[3] if len(nums) > 3 else 0

        #2: intensity  background
        nums = [float(s) for s in lines[1].split()]
        self.intensity, self.background = nums

        #3: maxLayer  nRoughSteps  nFitParam
        nums = [int(s) for s in lines[2].split()]
        maxLayer, self.roughness_steps, _ = nums

        #4-7: Qmin  Qmax  nQ (data points in a, b, c and d)
        # Note that we are only keeping the first one; for simulation all the
        # others should be the same, and for loading, the datafile will tell
        # us what Q to use.
        nums = [float(s) for s in lines[3].split()]
        self.Qmin, self.Qmax, self.num_Q = nums[0], nums[1], int(nums[2])

        #8: profile_type ('E' for error function, 'H' for tanh)
        self.roughness_profile = lines[7].strip()
        if self.roughness_profile not in ('E', 'H'):
            raise ValueError("Expected roughness profile type E or H")

        #9: active cross sections (usually 'abcd' or 'ABCD')
        self.active_xsec = lines[8].strip().lower()

        #10: data file (base name without suffix char such as test.refl)
        #11: output_file
        self.data_file = lines[9].strip()
        self.output_file = lines[10].strip()

        ## Layer information follows in sets of 3 lines x (nL+1).
        #Line next (1):  rho   depth   rough  mu
        #Line next (2):  mrho  mdepth  mrough (mrho is also known as phi)
        #Line next (3):  mtheta
        nL = maxLayer + 1
        layers = [[float(v) for v in " ".join(lines[11+3*i:11+3*(i+1)]).split()]
                  for i in range(nL)]

        A = np.array(layers)
        self.rho = A[:, 0]* (1e6/16/pi)
        self.irho = A[:, 3] * (1e6/2/self.wavelength)
        self.thickness = A[:, 1]
        self.roughness = A[:, 2]
        self.mrho = A[:, 4] * (1e6/16/pi)
        self.mtheta = A[:, 7]
        self.mthickness = A[:, 5]
        self.mroughness = A[:, 6]

        ## Fit information fills the remainder of the file
        #footer+1: P1 P2 P3 ...  (fit parameters)
        self.fitpars = [int(s) for s in lines[10+3*nL+1].split()]
        #footer+2 to end: constraints
        self.constraints = "".join(lines[10+3*nL+2:])

    def _write(self, fid):
        #1: wavelength  wavelength_dispersion  angular_divergence [aguide]
        fid.write("%g %g %g %g\n"%(self.wavelength, self.wavelength_dispersion,
                                   self.angular_divergence, self.guide_angle))

        #2: intensity  background
        fid.write("%g %g\n"%(self.intensity, self.background))

        #3: maxLayer  nRoughSteps  nFitParam
        fid.write("%d %d %d\n"%(len(self.rho)-1, self.roughness_steps,
                                len(self.fitpars)))

        #4-7: Qmin  Qmax  nQ (data points in a, b, c and d)
        fid.write("%g %g %d\n"%(self.Qmin, self.Qmax, self.num_Q))
        fid.write("%g %g %d\n"%(self.Qmin, self.Qmax, self.num_Q))
        fid.write("%g %g %d\n"%(self.Qmin, self.Qmax, self.num_Q))
        fid.write("%g %g %d\n"%(self.Qmin, self.Qmax, self.num_Q))

        #8: profile_type ('E' for error function, 'H' for tanh)
        fid.write("%s\n"%self.roughness_profile)

        #9: active cross sections (usually 'abcd' or 'ABCD')
        fid.write(" %s\n"%self.active_xsec)

        #10: data file (base name without suffix char such as test.refl)
        #11: output_file
        fid.write("%s\n%s\n"%(self.data_file, self.output_file))

        ## Layer information follows in sets of 3 lines x (nL+1).
        #Line next (1):  rho   depth   rough  mu
        #Line next (2):  mrho  mdepth  mrough (mrho is also known as phi)
        #Line next (3):  mtheta
        rho = self.rho*(16*pi/1e6)
        w, s = self.thickness, self.roughness
        wm = self.mthickness if self.mthickness is not None else self.thickness
        sm = self.mroughness if self.mroughness is not None else self.roughness
        if self.irho is not None:
            mu = self.irho*(2*self.wavelength/1e6)
        else:
            mu = np.zeros_like(rho)
        if self.mrho is not None:
            mrho = self.mrho * (16*pi/1e6)
        else:
            mrho = np.zeros_like(rho)
        if self.mtheta is not None:
            mtheta = self.mtheta
        else:
            mtheta = 270*np.ones_like(rho)
        for i in range(len(rho)):
            fid.write("%g %g %g %g\n%g %g %g\n%g\n"
                      %(rho[i], w[i], s[i], mu[i], mrho[i], wm[i], sm[i], mtheta[i]))

        ## Fit information fills the remainder of the file
        #footer+1: P1 P2 P3 ...  (fit parameters)
        fid.write(" ".join(str(p) for p in self.fitpars)+"\n")
        #footer+2 to end: constraints
        fid.write(self.constraints)
