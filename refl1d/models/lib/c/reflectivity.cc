/* This program is public domain */

/**
 *  reflectivity.
 */
#include <iostream>
#include <complex>
#include "reflcalc.h"

// Abeles matrix reflectivity calculation
static void
refl(const int layers,
     const double kz,
     const double depth[],
     const double sigma[],
     const double rho[],
     const double irho[],
     Cplx& R)
{
  const Cplx J(0,1);

  // Check that Q is not too close to zero.
  // For negative Q, reverse the layers.
  const double cutoff = 1e-10;
  int next,step;
  if (kz >= cutoff) {
    next=0;
    step=1;
  } else if (kz <= -cutoff) {
    next=layers-1;
    step=-1;
    sigma -= 1;
  } else {
    R = -1.;
    return;
  }

  // Since sqrt(1/4 * x) = sqrt(x)/2, I'm going to pull the 1/2 into the
  // sqrt to save a multiplication later.
  const double pi4=12.566370614359172e-6;        // 1e-6 * 4 pi
  const double kz_sq = kz*kz + pi4*rho[next];    // kz^2 + 4 pi Vrho
  Cplx k(fabs(kz));

  Cplx B11, B12, B21, B22;
  B11 = B22 = 1;
  B12 = B21 = 0;
#if 0
  std::cout << "kz: " << kz << std::endl;
#endif
  for (int i=0; i < layers-1; i++) {
    // The loop index is not the layer number because we may be reversing
    // the stack.  Instead, n is set to the incident layer (which may be
    // first or last) and incremented or decremented each time through.
    const Cplx k_next = sqrt(kz_sq - pi4*Cplx(rho[next+step],irho[next+step]));
    const Cplx F = (k-k_next)/(k+k_next)*exp(-2.*k*k_next*sigma[next]*sigma[next]);
    const Cplx M11 = (i>0 ? exp(J*k*depth[next]) : 1);
    const Cplx M22 = (i>0 ? exp(-J*k*depth[next]) : 1);
    const Cplx M21 = F*M11;
    const Cplx M12 = F*M22;

#if 0
    std::cout << next
        << " k:" << k << " k_next:" << k_next << " F:" << F
        << " d:" << depth[next] << " sigma:" << sigma[next]
        << " rho:" << rho[next] << " irho:" << irho[next]
        << std::endl;
#endif
    // Multiply existing layers B by new layer M
    // We have unrolled the matrix multiply for speed.
    Cplx C1, C2;
    C1 = B11*M11 + B21*M12;
    C2 = B11*M21 + B21*M22;
    B11 = C1;
    B21 = C2;
    C1 = B12*M11 + B22*M12;
    C2 = B12*M21 + B22*M22;
    B12 = C1;
    B22 = C2;
    next += step;
    k = k_next;
  }


  // And we are done.
  R = B12/B11;
}



extern "C" void
reflectivity_amplitude(const int    layers,
             const double depth[],
             const double sigma[],
             const double rho[],
             const double irho[],
             const int    points,
             const double kz[],
             const int    rho_index[],
             Cplx r[])
{
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i=0; i < points; i++) {
    const int offset = layers*(rho_index!=NULL ? rho_index[i] : 0);
    refl(layers, kz[i], depth, sigma, rho+offset, irho+offset, r[i]);
  }
}


/*************************************************************************/
// We need  a number of tests as follows:
// (note V=vacuum, S=substrate, n=interior layer n, r=reflectivity amplitude)
//    Check R matches precalculated r for profiles with:
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
//      then r(Q) = r'(-Q)
//    Check that identical layers can be merged:
//      r(Q) = r'(Q) when d(n)=0 and P' = P without layer n
//      r(Q) = r'(Q) when P=P', rho(n)=rho(n+1),mu(n)=mu(n+1),d(n)=d(n+1)=C
//    Check that algorithms are consistent
//      Parrat == matrix == magnetic matrix A with Qm = 0
//    Check that thick layers approximate substrate
//      r(Q) = r'(Q) when P' = P(1:n) and d(n)>>0, mu(n)>0
//      ?? d(n)>>0, mu(n)>0 averages to rho(S)=rho(n), mu(S)=0 over one repeat
//    Check that thick layers generate the appropriate fringes
//      |r| has repeats of period 2 pi / sum(d) for high Q
//      |r| has repeats of period 2 pi / d(n) for thick d(n) for high Q
//    Check values below the critical angle
//      critical angle at Q = sqrt(16 pi (rho(S)-rho(V)))
//      |r(Q)| = 1 for Q<Qc if mu = 0
//      |r(Q)| > 1 for Q<Qc if mu < 0
//      |r(Q)| < 1 for Q<Qc if mu > 0
//    ?? Check that phase increase is monotonic in Q
//    Check vacuum and substrate absorption
//      reject mu(S) < 0, ignore mu(V) != 0
//    Check large Q values against the Born approximation
//      |r(Q)| for P = rectangular barrier falls off like Q^-2
//      |r(Q)| for P = triangular barrier falls off like Q^-3
//      ?? |r(Q)| for P = gaussian barrier falls off like exp(-Q^2/2)
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
