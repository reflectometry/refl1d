// This program is public domain.

/** \file
 * Handle approximate roughness.
 */

#include <iostream>
#include <complex>
#include "reflcalc.h"

/** *****************************************************************
 * Based on: 
 *   Ankner JF, Majkrzak CF (1992) "Subsurface profile refinement for
 *   neutron reflectivity" in SPIE 1738 Neutron Optical Devices and 
 *   Applications, 260-269.
 *
 * extensions to handle approximate roughness due to Nevot and Croce.
 *
 * Note that the paper uses
 *    R = a^4 (R + F)/(RF - 1)
 * where
 *    a = exp(ikd/2)
 * but k = q/2, so
 *    a^4 = exp(iqd/4)^4 = exp(iqd)
 * which is what we use here.
 *
 * FIXME below the critical angle the above exponential increases
 * rapidly with depth unless absorption is positive.
 *
 * Note: precalculating S = 16*pi*rho[k] + 8i*pi*mu[k]/lambda saves
 * about 5-10% in execution speed, but it means that reflectivity
 * must be called with a work vector.
 ********************************************************************
 */
static void
refl(const int   layers,
     const double d[],
     const double sigma[],
     const double rho[],
     const double mu[],
     const double lambda,
     const double Q,
     refl_complex& R
     )
{
  const double Qcutoff = 1e-10;
  const refl_complex J(0,1);
  const double pi16=5.0265482457436690e1;
  const double pi8olambda = pi16/lambda/2.;
  refl_complex F, f, f_next;
  int n, r, step, vacuum, dr;

  if (Q >= Qcutoff) {
    n=r=layers-1;
    vacuum=0;
    step=-1;
  } else if (Q <= -Qcutoff) {
    n=0; r=1;
    vacuum=layers-1;
    step=1;
  } else {
    R = -1.;
    return;
  }

  /** substrate --- calculate the index of refraction.  Ignore depth
  * since the substrate is semi-infinite and we get no reflection
   * from the bottom interface.
  */
  const double Qsq = Q*Q;
  const double Qsqrel = Qsq + pi16*rho[vacuum];
  f_next = sqrt(refl_complex(Qsqrel-pi16*rho[n],pi8olambda*mu[n]));
  R = 0.;
  for (int i=2; i < layers; i++) {
    n += step; r += step;
    f = sqrt(refl_complex(Qsqrel-pi16*rho[n],pi8olambda*mu[n]));
    F = (f - f_next) / (f + f_next);
    /* Note: -0.5 rather than -2. because we are using Q = 2*k */
    if (sigma[r]>0) F *= exp(-0.5*f*f_next*sigma[r]*sigma[r]);
    R = exp(d[n]*J*f) * (R + F) / (R*F + 1.);
    f_next = f;
  }
 
  /** vacuum --- we've already accounted for the index of refraction
   * of the vacuum and we are measuring reflectivity relative to the 
   * top interface so we ignore absorption and depth.  This means that
   * S is 0 and the exponential is 1.
   */
  r += step;
  f = fabs(Q);
  F = (f-f_next) / (f+f_next);
  if (sigma[r] > 0) F *= exp(-0.5*f*f_next*sigma[r]*sigma[r]);
  R = (R + F) / (R*F + 1.);
}


extern "C" void 
reflrough_amplitude(const int layers,
                    const double depth[],
                    const double sigma[],
                    const double rho[],
                    const double mu[],
                    const double wavelength[],
                    const int points,
                    const double Q[],
                    refl_complex R[])
{
  for (int i=0; i < points; i++)
  refl(layers, depth, sigma, rho, mu, wavelength[i], Q[i], R[i]);
}



extern "C" void 
reflrough(const int layers,
          const double depth[],
          const double sigma[],
          const double rho[],
          const double mu[],
          const double wavelength[],
          const int points,
          const double Q[],
          double R[])
{
  for (int i=0; i < points; i++) {
      refl_complex amp;
      refl(layers, depth, sigma, rho, mu, wavelength[i], Q[i], amp);
      R[i] = real(amp * conj(amp));
  }
}
