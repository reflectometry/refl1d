/* This program is public domain. */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#ifdef SGI
#include <ieeefp.h>
#endif
#include "reflcalc.h"

/* What to do at the endpoints --- USE_TRUNCATED_NORMALIZATION will
 * avoid the assumption that the data is zero where it hasn't been
 * measured.
 */
#define USE_TRUNCATED_NORMALIZATION

/* Computed using extended precision with Octave's symbolic toolbox. */
#define PI4          12.56637061435917295385
#define PI_180        0.01745329251994329576
#define LN256         5.54517744447956247533
#define SQRT2         1.41421356237309504880
#define SQRT2PI       2.50662827463100050241

/* Choose the resolution limit based on relative contribution to resolution
 * rather than absolute contribution.  So if we ignore everything below,
 * e.g. 0.1% of the peak, that limit occurs when G(x)/G(0) = 0.001 for
 * gaussian G of width sima, or when x = sqrt(-2 sigma^2 log(0.001)). */
#define LOG_RESLIMIT -6.90775527898213703123

/** \file 
The resolution function returns the convolution of the reflectometry
curve with a Q-dependent gaussian.

We provide the following function:
   resolution(Nin, Qin, Rin, N, Q, dQ, R)  returns convolution
   resolution_padding(step,dQ)             returns \#points (see below)

where
   Nin is the number of theory points
   Qin,Rin are the computed theory points
   N is the number of Q points to calculate
   Q are the locations of the measured data points
   dQ are the width (sigma) of the convolution at each measured point
   R is the returned convolution.

Note that FWHM = sqrt(8 ln 2) dQ, so scale dQ appropriately.

The contribution of Q to a resolution of width dQo at point Qo is:

   p(Q) = 1/sqrt(2 pi dQo^2) exp ( (Q-Qo)^2/(2 dQo^2) )

We are approximating the convolution at Qo using a numerical
approximation to the integral over the measured points.  For 
efficiency, the integral is limited to p(Q_i)/p(0)>=0.001.  

Note that the function we are convoluting is falling off as Q^4.
That means the correct convolution should uniformly sample across
the entire width of the Gaussian.  This is not possible at the
end points unless you calculate the reflectivity beyond what is
strictly needed for the data. The function resolution_pad(dQ,step) 
returns the number of additional steps of size step required to 
go beyond this limit for the given width dQ.  This occurs when:

    (n*step)^2 < -2 dQ^2 * ln 0.001

The choice of sampling density is particularly important near the 
critical edge.  This is where the resolution calculation has the 
largest effect on the reflectivity curve. In one particular model, 
calculating every 0.001 rather than every 0.02 changed one value 
above the critical edge by 15%.  This is likely to be a problem for 
any system with a well defined critical edge.  The solution is to
compute the theory function over a finer mesh where the derivative
is changing rapidly.  For the critical edge, I have found a sampling
density of 0.005 to be good enough.

For systems involving thick layers, the theory function oscillates 
rapidly around the measured points.  This is a problem when the
period of the oscillation, 2 pi/d for total sample depth d, is on
the order of the width of the resolution function. This is true even 
for gradually changing profiles in materials with very high roughness
values.  In these systems, the theory function should be oversampled
around the measured points Q.  With a single thick layer, oversampling
can be limited to just one period.  With multiple thick layers,
oscillations will show interference patterns and it will be necessary 
to oversample uniformly between the measured points.  When oversampled
spacing is less than about 2 pi/7 d, it is possible to see aliasing
effects.  

FIXME is it better to use random sampling or strictly
regular spacing when you are undersampling?

===============================================================
*/

#undef USE_TRAPEZOID_RULE
#ifdef USE_TRAPEZOID_RULE
#warning This code does strange things with small sigma and large spacing
/* FIXME trapezoid performs very badly for large spacing unless
   we normalize to the unit width.  For very small sigma, the gaussian
   is a spike, but we are approximating it by a triangle so it is
   not surprising it works so poorly.  A slightly better solution is
   to use the inner limits rather than the outer limits, but this will
   still break down if the Q spacing is approximately equal to limit. 
   Best is to not use trapezoid.
*/

/* Trapezoid rule for numerical integration of convolution */
double 
convolve_point(const double Qin[], const double Rin[], int k, int n,
        double Qo, double limit, double sigma)
{
  const double two_sigma_sq = 2. * sigma * sigma;
  double z, Glo, RGlo, R, norm;
  
  z = Qo - Qin[k];
  Glo = exp(-z*z/two_sigma_sq);
  RGlo = Rin[k]*Glo;
  norm = R = 0.;
  while (++k < n) {
    /* Compute the next endpoint */
    const double zhi = Qo - Qin[k];
    const double Ghi = exp(-zhi*zhi/two_sigma_sq);
    const double RGhi = Rin[k] * Ghi;
    const double halfstep = 0.5*(Qin[k] - Qin[k-1]);
    
    /* Add the trapezoidal area. */
    norm += halfstep * (Ghi + Glo);
    R += halfstep * (RGhi + RGlo);
    
    /* Save the endpoint for next trapezoid. */
    Glo = Ghi;
    RGlo = RGhi;
    
    /* Check if we've calculated far enough */
    if (Qin[k] >= Qo+limit) break;
  }

  /* Scale to area of the linear spline distribution we actually used. */
  return R / norm;

  /* Scale to gaussian of unit area */
  /* Fails badly for small sigma or large Q steps---do not use. */
  /* return R / sigma*SQRT2PI; */
}

#else /* !USE_TRAPEZOID_RULE */

/* Analytic convolution of gaussian with linear spline */
/* More expensive but more reliable */
double 
convolve_point(const double Qin[], const double Rin[], int k, int n,
               double Qo, double limit, double sigma)
{
  const double two_sigma_sq = 2. * sigma * sigma;
  double z, Glo, erflo, erfmin, R;
  
  z = Qo - Qin[k];
  Glo = exp(-z*z/two_sigma_sq);
  erfmin = erflo = erf(-z/(SQRT2*sigma));
  R = 0.;
  /* printf("%5.3f: (%5.3f,%11.5g)",Qo,Qin[k],Rin[k]); */
  while (++k < n) {
  	/* No additional contribution from duplicate points. */
  	if (Qin[k] == Qin[k-1]) continue;
 
    /* Compute the next endpoint */
    const double zhi = Qo - Qin[k];
    const double Ghi = exp(-zhi*zhi/two_sigma_sq);
    const double erfhi = erf(-zhi/(SQRT2*sigma));
    const double m = (Rin[k]-Rin[k-1])/(Qin[k]-Qin[k-1]);
    const double b = Rin[k] - m * Qin[k];

    /* Add the integrals. */
    R += 0.5*(m*Qo+b)*(erfhi-erflo) - sigma/SQRT2PI*m*(Ghi-Glo);

    /* Debug computation failures. */
    /*
    if isnan(R) {
    	printf("NaN from %d: zhi=%g, Ghi=%g, erfhi=%g, m=%g, b=%g\n",
    	       k,zhi,Ghi,erfhi,m,b);
    }
    */
    
    /* Save the endpoint for next trapezoid. */
    Glo = Ghi;
    erflo = erfhi;
    
    /* Check if we've calculated far enough */
    if (Qin[k] >= Qo+limit) break;
  }
  /* printf(" (%5.3f,%11.5g)",Qin[k<n?k:n-1],Rin[k<n?k:n-1]); */

#ifdef USE_TRUNCATED_NORMALIZATION
  /* Normalize by the area of the truncated gaussian */
  /* At this point erflo = erfmax */
  /* printf ("---> %11.5g\n",2*R/(erflo-erfmin)); */
  return 2 * R / (erflo - erfmin);
#else
  /* Return unnormalized (we used a gaussian of unit area) */
  /* printf ("---> %11.5g\n",R); */
  return R;
#endif
}

#endif /* !USE_TRAPEZOID_RULE */

void
resolution(int Nin, const double Qin[], const double Rin[],
           int N, const double Q[], const double dQ[], double R[])
{
  int lo,out;

  /* FIXME fails if Qin are not sorted; slow if Q not sorted */
  assert(Nin>1);

  /* Scan through all Q values to be calculated */
  lo = 0;
  for (out=0; out < N; out++) {
    /* width of resolution window for Q is w = 2 dQ^2. */
    const double sigma = dQ[out];
    const double Qo = Q[out];
    const double limit = sqrt(-2.*sigma*sigma* LOG_RESLIMIT);

    /* if (out%20==0) printf("%d: Q,dQ = %g,%g\n",out,Qo,sigma); */

    /* Line up the left edge of the convolution window */
    /* It is probably forward from the current position, */
    /* but if the next dQ is a lot higher than the current */
    /* dQ or if the Q are not sorted, then it may be before */
    /* the current position. */
    /* FIXME verify that the convolution window is just right */
    while (lo < Nin-1 && Qin[lo] < Qo-limit) lo++;
    while (lo > 0 && Qin[lo] > Qo-limit) lo--;

    /* Special handling to avoid 0/0 for w=0. */
    if (sigma > 0.) {
      R[out] = convolve_point(Qin,Rin,lo,Nin,Qo,limit,sigma);
    } else if (lo < Nin-1) {
      /* Linear interpolation */
      double m = (Rin[lo+1]-Rin[lo])/(Qin[lo+1]-Qin[lo]);
      double b = Rin[lo] - m*Qin[lo];
      R[out] = m*Qo + b;
    } else if (lo > 0) {
      /* Linear extrapolation */
      double m = (Rin[lo]-Rin[lo-1])/(Qin[lo]-Qin[lo-1]);
      double b = Rin[lo] - m*Qin[lo];
      R[out] = m*Qo + b;
    } else {
      /* Can't happen because there is more than one point in Qin. */
      assert(Nin>1);
    }
  }

}

/* ===================================================================

The FWHM of the resolution function for each Q is given as:
   w = |Q| sqrt( (dL/L)^2 + (dT/tan(T))^2 )
where
   T = arcsin ( |Q| L / 4 pi)

Return dQ as standard deviation:
   dQ = w / sqrt(8 ln(2))
  
We provide the following functions:
   resolution_varying(L, dL/L, dT/T, n, Q, dQ)  returns dQ
   resolution_fixed(L, dL/L, dT, n, Q, dQ)      returns dQ
   resolution_dQoQ(dQ/Q, n Q, dQ)               returns dQ
   resolution_dT(s1,s2,d)                       returns dT
   resolution_dToT(s1,s2,d,A3)                  returns dT/T
where
   L,dL is the wavelength and divergence in angstroms
   T,dT is the incident angle theta and angular divergence in radians
   Q,dQ is the reciprocal distance and divergence in inverse angstroms
   A3 in the incident angle theta in degrees
   s1,s2 are the slit openings
   d is the separateion between the slits in the same units as s1,s2
  
For a particular wavelength L, the angular divergence dT is 
approximated by a gaussian with FWHM:
           
   dT = s/d
  
with s being the opening of slits s1 and s2 and d the distance 
between slits.
   
This formula for angular divergence dT is a small angle approximation 
which assumes an isotropic field of neutrons before the first slit.  
In these conditions, the angular distribution is triangular of 
width 2 dT, so the FWHM is dT.

More precisely, the maximum divergence is Amax = arctan(s/d) ~ s/d in 
either direction.  The number of neutrons at an angle |A| less than Amax 
is given by the isotropic intensity I and the details of the slit:
   I(A) = I d ( s/d - tan |A| ) cos |A|
You can derive this from simple geometry drawing the two pairs of
slits and connecting the extreme edges:
  
      S1     S2  *beam
      |      |***
           **C s2/2
   -----***---------
     *A*     B s1/2
   ** |--d---|
  
The triangle CAB is a right triangle with opposite (s1+s2)/2 and
adjacent d, which for s1==s2 has angle Amax = arctan(s/d).  Consider
an angle A < Amax given by triangle BAC*, with C* between B and C.
The transmitted beam for angle A is the area between AC* and a
parallel line A*C which goes through point C.  Then beam intensity
for this angle I(A) = I times the perpendicular distance between 
these two parallel lines. As angle A decreases and C* approaches B,
the beam width increases up to a maximum corresponding to the slit
opening.  The situation is symmetric for angles A < 0, leading to
a symmetric distribution which is approximately triangular up to
about s/d = 0.5.  This is well above the range of values we will see.
 
Note that the usual transformation, sigma = FWHM / sqrt(8 ln 2), 
overestimates the width of the gaussian convolution function by 
about 4% compared to the expected variance of a triangular distribution.  
Strictly speaking, using a Gaussian resolution function is not correct, 
but when combined with wavelength dispersion the result is close 
enough to Gaussian for our purposes.
  
If the slits are different widths use:
   s = (s1+s2)/2.  
The precise analysis is more difficult because the distribution is a 
truncated triangle.  For a range of angles, the maximum beam width
through one set of slits passes unobstructed through the second set of
slits.  For very large differences in slit width, this leads to an
approximately rectangular distribution being assigned a FWHM of about
half its width, which underestimates the width of the gaussian 
convolution by as much as 35%.  The instrument is never operated in
this region.
  
For small samples use:
  
   dT = (s2 + h)/(2 d2)
  
where h is the width of the sample and d2 is the distance from sample
to slit s2.  
  
FIXME make sure the estimate is reasonable given small samples
and usual slits.
  
Preforming a stable calculation of dQ requires some tricks.  From
above we have:
   T = asin ( |Q| L / 4 pi)
   dQ/Q = sqrt( (dL/L)^2 + (dT/tan(T))^2 )
Note:  
   tan(asin(x)) = x/sqrt(1 - x^2)
so:
   dQ/Q = sqrt( (dL/L)^2 + dT^2 ((4 pi / Q L)^2 - 1 ) )
  
This blows up as Q -> 0, so bring Q under the sqrt:
   dQ = sqrt( Q^2 (dL/L)^2 + dT^2 ((4 pi / L)^2 - Q^2) )

This result is stable even for Q = 0. 

============================================================ */

void
resolution_varying(double L, double dLoL, double dToT, 
                   int n, const double Q[], double dQ[])
{
  /* Given:
       T = asin(Q L / 4 pi)
       dT = T dToT
       dLoL = dL/L
       dQ = sqrt( Q^2 (dL/L)^2 + dT^2 ((4 pi / L)^2 - Q^2) )
     substitute in dLoL and dToT
       dQ = sqrt( Q^2 dLoL^2 + T^2 dToT^2 (4 pi / L)^2 - T^2 dToT^2 Q^2 )
          = sqrt( T^2 (4 pi dToT/L)^2 + Q^2 ( dLoL^2 - T^2 dToT^2) )
   = sqrt( S1^2 T^2 + S2^2 Q^2 - S3^2 T^2 Q^2 )
     Divide by sqrt(8ln2) for sigma rather than FWHM.  This is
     equivalent to dividing S1^2, S2^2 and S3^2 by 8ln2, which 
     is what I've done in the definitions below.
  */
  const double Lo4pi = L / PI4;
  const double S1sq = dToT * dToT / ( Lo4pi * Lo4pi * LN256);
  const double S2sq = dLoL * dLoL / LN256;
  const double S3sq = dToT * dToT / LN256;
  int j;

  for (j=0; j < n; j++) {
    const double T = asin(Q[j] * Lo4pi);
    const double Tsq = T*T;
    const double Qsq = Q[j]*Q[j];
    dQ[j] = sqrt( S1sq * Tsq + S2sq * Qsq - S3sq * Tsq * Qsq );
  }
}

void
resolution_fixed(double L, double dLoL, double dT, 
                 int n, const double Q[], double dQ[])
{
#if 1
  /* Given:
       dT, L, Q
       T = asin(Q L / 4 pi)
       dLoL = dL/L
     Calculate:
       dQ = sqrt( Q^2 (dL/L)^2 + dT^2 ((4 pi / L)^2 - Q^2) )
     Substitute in dLoL
       dQ = sqrt( Q^2 dLoL^2 + dT^2 (4 pi / L)^2 - dT^2 Q^2 )
          = sqrt( (4 pi dT/L)^2 + Q^2 ( dLoL^2 - dT^2) )
   = sqrt( S1 + S2 Q^2 )
     Divide by sqrt(8ln2) for sigma rather than FWHM.  This is
     equivalent to dividing S1 and S2 by 8ln2, which is what I've
     done in the definitions below.
  */
  const double pi4dToL = (PI4 * dT / L);
  const double S1 = pi4dToL * pi4dToL / LN256;
  const double S2 = (dLoL * dLoL - dT * dT) / LN256;
  int j;

  for (j=0; j < n; j++) dQ[j] = sqrt( S1 + S2 * Q[j]*Q[j]);
#else
  /* mlayer resolution calculation 
   * Simple adding rather than adding in quadrature.
   */
  int j;
  for (j=0; j < n; j++) 
    dQ[j] = (fabs(Q[j]) * dLoL + 4. * M_PI * dT) / L / sqrt(LN256);
#endif
}

void
resolution_dQoQ(double dQoQ, int n, const double Q[], double dQ[])
{
  int j;
  for (j=0; j < n; j++) dQ[j] = dQoQ * Q[j];
}



double resolution_dT(double s1,double s2,double d)
{
  return (s1+s2)/(2*d);
}

double resolution_dToT(double s1,double s2,double d,double A3)
{
  return resolution_dT(s1,s2,d)/(A3*PI_180);
}

/* Compute the number of steps of padding needed to compute the resolution
 * at Qo given a step size of d and a resolution width of dQ.  See comments
 * on resolution_width for a definition of dQ. The resolution extends out
 * to a value of 0.1% of the value at the peak for RESLIMIT=0.001.
 */
int
resolution_padding(double step, double sigma)
{
  const double limit = sqrt(-2.*sigma*sigma * LOG_RESLIMIT);
  return ceil(limit/step);
}
