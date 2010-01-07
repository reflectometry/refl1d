/* This program is public domain */

/**
 *  reflectivity.
 */

#undef TRACE
#ifdef TRACE
# include <iostream>
#endif
#include <complex>
#include "reflcalc.h"


#if defined(USE_PARRAT)
#warning Using Parrat
// based on:
//   Ankner JF, Majkrzak CF (1992) "Subsurface profile refinement for
//   neutron reflectivity" in SPIE 1738 Neutron Optical Devices and
//   Applications, 260-269.
//
// Note that the paper uses
//    R = a^4 (R + F)/(RF - 1)
// where
//    a = exp(ikd/2)
// but k = q/2, so
//    a^4 = exp(iqd/4)^4 = exp(iqd)
// which is what we use here.
//
// FIXME below the critical angle the above exponential increases
// rapidly with depth unless absorption is positive.  If this is a problem
// use the matrix formalism below.
//
// Note: precalculating S = 16*pi*rho[k] + 8i*pi*mu[k]/lambda saves
// about 5-10% in execution speed, but it means that reflectivity
// must be called with a work vector.

static void
refl(const int layers,
     const double Q,
     const double depth[],
     const double rho[],
     const double mu[],
     const double wavelength,
     refl_complex& R
   )
{
  const double Qcutoff = 1e-10;
  const refl_complex J(0,1);
  const double pi16=5.0265482457436690e1;
  const double pi8owavelength = (pi16/2.0)/wavelength;
  refl_complex F, f, f_next;
  int n, step, vacuum;

  if (Q >= Qcutoff) {
    n=layers-1;
    vacuum=0;
    step=-1;
  } else if (Q <= -Qcutoff) {
    n=0;
    vacuum=layers-1;
    step=1;
  } else {
    R = -1.;
    return;
  }

  // substrate --- calculate the index of refraction.  Ignore depth
  // since the substrate is semi-infinite and we get no reflection
  // from the bottom interface.
  const double Qsqrel = Q*Q + pi16*rho[vacuum];
  f_next = sqrt(refl_complex(Qsqrel-pi16*rho[n],pi8owavelength*mu[n]));
  R = 0.;
  for (int i=2; i < layers; i++) {
    n += step;
    f = sqrt(refl_complex(Qsqrel-pi16*rho[n],pi8owavelength*mu[n]));
    F = (f - f_next) / (f + f_next);
    R = exp(depth[n]*J*f) * (R + F) / (R*F + 1.);
    f_next = f;
  }
  // vacuum --- we've already accounted for the index of refraction
  // of the vacuum and we are measuring reflectivity relative to the
  // top interface so we ignore absorption and depth.  This means that
  // S is 0 and the exponential is 1.
  f = fabs(Q);
  F = (f-f_next) / (f+f_next);
  R = (R + F) / (R*F + 1.);
}

#else /* !USE_PARRAT */


#ifdef TRACE
int trace = 0;
#endif

// Matrix formalism implementation
// Modification of C.F. Majrkzak's progam gepore.f for calculating
// reflectivities of four polarization states of neutron reflectivity data.
static void
refl(const int layers,
     const double Q,
     const double depth[],
     const double rho[],
     const double mu[],
     const double wavelength,
     refl_complex& R)
{
  const refl_complex J(0,1);

  // Check that Q is not too close to zero.
  // For negative Q, reverse the layers.
  const double Qcutoff = 1e-10;
  int n,step;
  if (Q >= Qcutoff) {
    n=0;
    step=1;
  } else if (Q <= -Qcutoff) {
    n=layers-1;
    step=-1;
  } else {
    R = -1.;
    return;
  }

  // Since sqrt(1/4 * x) = sqrt(x)/2, I'm going to pull the 1/2 into the
  // sqrt to save a multiplication later.
  const double pi4=1.2566370614359172e1;            //4 pi
  const double pi2owavelength = 0.5*pi4/wavelength; //2 pi / wavelength
  const double Qsqrel = 0.25*Q*Q + pi4*rho[n];      //0.25 * (Q^2 + 16 pi Vrho)

  refl_complex B11, B12, B21, B22;
  B11 = B22 = 1.0;
  B12 = B21 = 0.0;
  for (int i=2; i < layers; i++) {
    // Loop starts at 2 because vacuum and substrate are in the loop.  The
    // loop index is not the layer number because we may be going backward
    // or forward.  Instead, n is set to the incident layer (which may be
    // first or last) and incremented or decremented each time through.
    n += step;
    // Given
    //
    //    Qc^2(L) = 16 pi rho(L)
    //    S1 = 1/2 sqrt(Qc^2(L) - Qc^2(vacuum) - Q^2 - 8i pi mu(L)/wavelength),
    //
    // compute the matrix A
    //
    //           1     (  cosh(d S1)   sinh(d S1)/S1 )
    //    A = -------- (                             )
    //        exp(a d) ( sinh(d S1)*S1   cosh(d S1)  )
    //
    // where a is the magnitude of the real part of S1.
    //
    // The scale factor
    //
    //     H=exp(a d)
    //
    // keeps the calculation stable even for large thickness d. We can use
    // any scale factor we want since it cancels later when we calculate
    // the reflectivity  R = V/U.
    const refl_complex S1 = sqrt(refl_complex(pi4*rho[n]-Qsqrel,
                       -pi2owavelength*mu[n]));
    // I'm unrolling the sinh/cosh computations because that allows me to
    // save some exponentials and trig functions.  I'm also wrapping in the
    // division by H=exp(a d) since this is just arithmetic on the arg to
    // the exp function, and I save another exponential.  Depending on the
    // sign of a, which is negative below the critical angle, we need to
    // divide either exp(ad) or exp(-ad). For the (exp(a)+exp(-a))/2 term
    // it doesn't matter which since both yield (1+exp(-2|a|))/2, but for
    // the (exp(a)-exp(-a))/2 term this will change the sign.
#ifdef TRACE
    const double epa  = exp(real(S1)*depth[n]);
    const double rexp = (epa + 1./epa)/2.;
    const double rexm = (epa - 1./epa)/2.;
#else
    const double em2a = exp(-2.*fabs(real(S1))*depth[n]);
    const double rexp = (1.+em2a)/2.;
    const double rexm = ( real(S1)>0. ? (1.-em2a)/2. : (em2a-1.)/2. );
#endif // !TRACE

#if defined(HAVE_SINCOS)
    double costheta, sintheta;
    sincos(imag(S1)*depth[n],&sintheta,&costheta);
#else
    const double sintheta = sin(imag(S1)*depth[n]);
    const double costheta = cos(imag(S1)*depth[n]);
#endif // !HAVE_SINCOS

    const refl_complex Adiag(rexp*costheta,rexm*sintheta); // = coshS1
    const refl_complex sinhS1(rexm*costheta,rexp*sintheta);
    const refl_complex A12 = sinhS1/S1;
    const refl_complex A21 = sinhS1*S1;

    // Multiply A by existing layers B
    // We have unrolled the matrix multiply for speed.
    refl_complex C1, C2;
    C1 = (Adiag*B11 + A12*B21);
    C2 = (A21*B11 + Adiag*B21);
    B11 = C1;
    B21 = C2;
    C1 = (Adiag*B12 + A12*B22);
    C2 = (A21*B12 + Adiag*B22);
    B12 = C1;
    B22 = C2;

#ifdef TRACE
    if (trace) {
      std::cout << "f="<<S1 << std::endl;
      std::cout << "A=["<<Adiag<<" "<<A12<<" "<<A21<<"]" << std::endl;
      std::cout << "B=["<<B11<<" "<<B12<<" "<<B21<<"]" << std::endl;
    }
#endif
  }

  // Use corrected versions of X,Y,ZI, and ZS to account for effect
  // of incident and substrate media.  Remember that we have already
  // accounted for incident media in Qsqrel, and that we are computing
  // sqrt(S1/4) instead of 1/2 sqrt(S1) for substrate parameters S1.
  n+=step;
  const refl_complex ZS = J*sqrt(refl_complex(Qsqrel-pi4*rho[n],
                       pi2owavelength*mu[n]));
  const refl_complex ZI = J*fabs(0.5*Q);

  // Save a few more multiplies by gathering the following:
  //   X=-1; Y = ZI*ZS;
  //   U = (Y*B12 - ZI*B22) + (ZS*B11 + X*B21)
  //   V = (Y*B12 - ZI*B22) - (ZS*B11 + X*B21)
  // into
  //   U = a + b
  //   V = a - b
  const refl_complex a = ZI*ZS*B12 - ZI*B22;
  const refl_complex b = ZS*B11 - B21;
  const refl_complex U = a - b;
  const refl_complex V = a + b;

#ifdef TRACE
  if (trace) {
    std::cout << "fi=" << ZI << ", fs=" << ZS << std::endl;
    std::cout << "a=" << a << ", b=" << b << std::endl;
  }
#endif

  // And we are done.
  R = V/U;
}
#endif /* !USE_PARRAT */



extern "C" void
reflectivity_amplitude(const int    layers,
                       const double depth[],
                       const double rho[],
                       const double mu[],
                       const double wavelength[],
                       const int    points,
                       const double Q[],
                       refl_complex R[])
{
  for (int i=0; i < points; i++)
    refl(layers, Q[i], depth, rho, mu, wavelength[i], R[i] );
}






/*************************************************************************/
// We need  a number of tests as follows:
// (note V=vacuum, S=substrate, n=interior layer n, R=reflectivity amplitude)
//    Check R matches precalculated R for profiles with:
//      rho(n)=rho(V) for rho(V)=0 and rho(V)!=0
//      rho(n)=0
//      rho(S)=0 and rho(S)!=0
//      rho(V)=0 and rho(V)!=0
//      rho(V)=rho(S) for rho(V)=0 and rho(V)!=0
//      rho(n)<0
//      rho(n)>0
//    Check that we can reverse profiles:
//      Assume no absorption in the substrate => mu(S)=mu(V)=0
//      Given mu'=reverse(mu), d'=reverse(d), rho'=reverse(rho)-rho(S)+rho(V)
//      then R(Q) = R'(-Q)
//    Check that identical layers can be merged:
//      R(Q) = R'(Q) when d(n)=0 and P' = P without layer n
//      R(Q) = R'(Q) when P=P', rho(n)=rho(n+1),mu(n)=mu(n+1),d(n)=d(n+1)=C
//    Check that algorithms are consistent
//      Parrat == matrix == magnetic matrix A with Qm = 0
//    Check that thick layers approximate substrate
//      R(Q) = R'(Q) when P' = P(1:n) and d(n)>>0, mu(n)>0
//      ?? d(n)>>0, mu(n)>0 averages to rho(S)=rho(n), mu(S)=0 over one repeat
//    Check that thick layers generate the appropriate fringes
//      |R| has repeats of period 2 pi / sum(d) for high Q
//      |R| has repeats of period 2 pi / d(n) for thick d(n) for high Q
//    Check values below the critical angle
//      critical angle at Q = sqrt(16 pi (rho(S)-rho(V)))
//      |R(Q)| = 1 for Q<Qc if mu = 0
//      |R(Q)| > 1 for Q<Qc if mu < 0
//      |R(Q)| < 1 for Q<Qc if mu > 0
//    ?? Check that phase increase is monotonic in Q
//    Check vacuum and substrate absorption
//      reject mu(S) < 0, ignore mu(V) != 0
//    Check large Q values against the Born approximation
//      |R(Q)| for P = rectangular barrier falls off like Q^-2
//      |R(Q)| for P = triangular barrier falls off like Q^-3
//      ?? |R(Q)| for P = gaussian barrier falls off like exp(-Q^2/2)
//      In octave:
/*
          n=4000;
   wb=1e-7*boxcar(2*n-1)(1:n); wb(1)=0;
   wt=1e-7*bartlett(2*n-1)(1:n);
   wg=1e-7*gausswin(2*n-1,10)(1:n);
   d=1000*ones(n,1)/n;
   Q=logspace(-1,1,200)';
   Rb=abs(reflectivity(Q,[d,wb],5));
   Rt=abs(reflectivity(Q,[d,wt],5));
   Rg=abs(reflectivity(Q,[d,wg],5));
   wpolyfit(log10(Q),log10(Rb),1);       % should be -2
   wpolyfit(log10(Q),log10(Rt),1);       % should be -3
   wpolyfit(log10(Q),log10(-2*log(Rg)),1); % should be 2
*/
//    Check alternate return values
//      reflectivity == abs(reflectivity_amplitude)^2
//      reflectivity_real == real(reflectivity_amplitude)
//    Compare reflectivity to exact analytic expression
//      e.g., Zhang and Lynn, "Analytic calculation of polarized neutorn
//      reflectivity from superconductors", PhysRevB 48(21) 1993

// $Id: reflectivity.cc 251 2007-06-15 17:10:19Z ziwen $
