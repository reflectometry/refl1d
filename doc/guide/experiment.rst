.. _experiment-guide:

*******************
Experiment
*******************

.. contents:: :local:

The :class:`Experiment <refl1d.experiment.Experiment>` object links a
`sample <sample-guide>`_ with an experimental `probe <data-guide>`_.
The probe defines the Q values and the resolution of the individual
measurements, and returns the scattering factors associated with the
different materials in the sample.

Because our models allow representation based on composition, it is no
longer trivial to compute the reflectivity from the model.  We now have
to look up the effective scattering density based on the probe type and
probe energy.  You've already seen this in :ref:`new-layers`:
the render method for the layer requires the probe to look up the material
scattering factors.


Direct Calculation
==================

Rather than using :class:`Stack <refl1d.model.Stack`,
:class:`Probe <refl1d.probe.Probe>` and
class:`Experiment <refl1d.experiment.Experiment`,
we  can compute reflectivities directly with the functions in
:mod:`refl1d.reflectivity`.  These routines provide the raw
calculation engines for the optical matrix formalism, converting
microslab models of the sample into complex reflectivity amplitudes,
and convolving the resulting reflectivity with the instrument resolution.

The following performs a complete calculation for a silicon
substrate with 5 |Ang| roughness using neutrons.  The theory is sampled
at intervals of 0.001, which is convolved with a 1% $\Delta Q/Q$ resolution
function to yield reflectivities at intervals of 0.01.

::

    >>> from numpy import arange
    >>> from refl1d.reflectivity import reflectivity_amplitude as reflamp
    >>> from refl1d.reflectivity import convolve
    >>> Qin = arange(0,0.21,0.001)
    >>> w,rho,irho,sigma = zip((0,2.07,0,5),(0,0,0,0))
    >>> # the last layer has no interface
    >>> r = reflamp(kz=Qin/2, depth=w, rho=rho, irho=irho, sigma=sigma[:-1])
    >>> Rin = (r*r.conj()).real
    >>> Q = arange(0,0.2,0.01)
    >>> dQ = Q*0.01 # resolution dQ/Q = 0.01
    >>> R = convolve(Qin, Rin, Q, dQ)
    >>> print("\n".join("Q: %.2f  R: %.5e"%(Qi,Ri) for Qi,Ri in zip(Q,R)))
    Q: 0.00  R: 1.00000e+00
    Q: 0.01  R: 3.11332e-02
    Q: 0.02  R: 3.30684e-03
    ...
    Q: 0.19  R: 2.10084e-07
