// This program is public domain.

// Force DEBUG mode with assertions tested.
//#ifdef NDEBUG
//#undef NDEBUG
//#endif
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <algorithm>

#define GREEDY
#include <cassert>

#define Z_EPS 1e-6

typedef double FullLayer[6];
extern "C"
int
align_magnetic(int nlayers, double d[], double sigma[], double rho[], double irho[],
               int nlayersM, double dM[], double sigmaM[], double rhoM[], double thetaM[],
               int noutput, double output_flat[])
{
  // ignoring thickness d on the first and last layers
  // ignoring interface width sigma on the last layer
  // making sure there are at least two layers
  assert(nlayers>1);
  assert(nlayersM>1);

  FullLayer *output = (FullLayer *)output_flat;
  int magnetic = 0; // current magnetic layer index
  int nuclear = 0; // current nuclear layer index
  double z = 0.; // current interface depth
  double next_z = 0.; // next nuclear interface
  double next_zM = 0.; // next magnetic interface
  //int active = 3; // active interfaces, active&0x1 for nuclear, active&0x2 for magnetic
  int k = 0;  // current output layer index
  while (1) { // repeat over all nuclear/magnetic layers
    if (k == noutput) return -1;  // exceeds capacity of output
    assert(nuclear < nlayers);
    assert(magnetic < nlayersM);
    //printf("%d: %d %d %g %g %g\n", k, nuclear, magnetic, z, next_z, next_zM);
    //printf("%g %g %g %g\n", rho[nuclear], irho[nuclear], rhoM[magnetic], thetaM[magnetic]);
    //printf("%g %g %g %g\n", d[nuclear], sigma[nuclear], dM[magnetic], sigmaM[magnetic]);

    // Set the scattering strength using the current parameters
    output[k][2] = rho[nuclear];
    output[k][3] = irho[nuclear];
    output[k][4] = rhoM[magnetic];
    output[k][5] = thetaM[magnetic];

    // Check if we are at the last layer for both nuclear and magnetic
    // If so set thickness and interface width to zero.  We are doing a
    // center of the loop exit in order to make sure that the final layer
    // is added.
    if (magnetic == nlayersM-1 && nuclear == nlayers-1) {
      output[k][0] = 0.;
      output[k][1] = 0.;
      k++;
      break;
    }

    // Determine if we are adding the nuclear or the magnetic interface next,
    // or possibly both.  The order of the conditions is important.
    //
    // Note: the final value for next_z/next_zM is not defined.  Rather than
    // checking if we are on the last layer we simply add the value of the
    // last thickness to z, which may be 0, nan, inf, or anything else.  This
    // doesn't affect the algorithm since we don't look at next_z when we are
    // on the final nuclear layer or next_zM when we are on the final magnetic
    // layer.
    //
    // Note: averaging nearly aligned interfaces can lead to negative thickness
    // Consider nuc = [1-a, 0, 1] and mag = [1+a, 1, 1] for 2a < Z_EPS.
    // On the first step we set next_z to 1-a, next_zM to 1+a and z to the
    // average of 1-a and 1+a, which is 1.  On the second step next_z is
    // still 1-a, so the thickness next_z - z = -a. Since a is tiny we can just
    // pretend that -a == zero by setting thickness to fmax(next_z - z, 0.0).
    //
    if (nuclear == nlayers-1) {
      // No more nuclear layers... play out the remaining magnetic layers.
      output[k][0] = std::max(next_zM - z, 0.0);
      output[k][1] = sigmaM[magnetic];
      next_zM += dM[++magnetic];
    } else if (magnetic == nlayersM-1) {
      // No more magnetic layers... play out the remaining nuclear layers.
      output[k][0] = std::max(next_z - z, 0.0);
      output[k][1] = sigma[nuclear];
      next_z += d[++nuclear];
    } else if (fabs(next_z - next_zM) < Z_EPS && fabs(sigma[nuclear]-sigmaM[magnetic]) < Z_EPS) {
      // Matching nuclear/magnetic boundary, with almost identical interfaces.
      // Increment both nuclear and magnetic layers.
      output[k][0] = std::max(0.5*(next_z + next_zM) - z, 0.0);
      output[k][1] = 0.5*(sigma[nuclear] + sigmaM[magnetic]);
      next_z += d[++nuclear];
      next_zM += dM[++magnetic];
    } else if (next_zM < next_z) {
      // Magnetic boundary comes before nuclear boundary, so increment magnetic.
      output[k][0] = std::max(next_zM - z, 0.0);
      output[k][1] = sigmaM[magnetic];
      next_zM += dM[++magnetic];
    } else {
      // Nuclear boundary comes before magnetic boundary
      // OR nuclear and magnetic boundaries match but interfaces are different.
      // so increment nuclear.
      output[k][0] = std::max(next_z - z, 0.0);
      output[k][1] = sigma[nuclear];
      next_z += d[++nuclear];
    }
    z += output[k][0];
    k++;
  }
  return k;
}

extern "C"
int
contract_by_step(int n, double d[], double sigma[],
                 double rho[], double irho[], double dh)
{
  double dz,rholeft,rhoarea,irholeft,irhoarea;
  int newi = 0;
  int i;

  dz = d[0];
  rholeft=rho[0];
  irholeft=irho[0];
  rhoarea = dz*rholeft;
  irhoarea = dz*irholeft;
  for (i=1; i < n; i++) {
    if (sigma[i] != 0.
        && fabs(rholeft-rho[i]) < dh
        && fabs(irholeft-irho[i]) < dh) {
      dz += d[i];
      rhoarea += d[i]*rho[i];
      irhoarea += d[i]*irho[i];
    } else {
      d[newi] = dz;
      if (newi > 0) {
	rho[newi] = rhoarea/dz;
	irho[newi] = irhoarea/dz;
      }
      newi++;

      dz = d[i];
      rholeft=rho[i];
      irholeft=irho[i];
      rhoarea = dz*rholeft;
      irhoarea = dz*irholeft;
    }
  }
  d[newi]=dz;
  rho[newi]=rho[n-1];
  irho[newi]=irho[n-1];
  return ++newi;
}


#ifdef GREEDY
extern "C"
int
contract_by_area(int n, double d[], double sigma[],
                 double rho[], double irho[], double dA)
{
  double dz;
  double rholo, rhohi, rhoarea;
  double irholo, irhohi, irhoarea;
  int i, newi;
  i=newi=1; /* Skip the substrate */
  while (i < n) {

    /* Get ready for the next layer */
    /* Accumulation of the first row happens in the inner loop */
    dz = rhoarea = irhoarea = 0.;
    rholo=rhohi=rho[i];
    irholo=irhohi=irho[i];

    /* Accumulate slices into layer */
    for (;;) {
      assert(i < n);
      /* Accumulate next slice */
      dz += d[i];
      rhoarea+=d[i]*rho[i];
      irhoarea+=d[i]*irho[i];

      /* If no more slices or sigma != 0, break immediately */
      if (++i == n || sigma[i-1] != 0.) break;

      /* If next slice won't fit, break */
      if (rho[i] < rholo) rholo = rho[i];
      if (rho[i] > rhohi) rhohi = rho[i];
      if ((rhohi-rholo)*(dz+d[i]) > dA) break;

      if (irho[i] < irholo) irholo = irho[i];
      if (irho[i] > irhohi) irhohi = irho[i];
      if ((irhohi-irholo)*(dz+d[i]) > dA) break;
    }

    /* dz is only going to be zero if there is a forced break due to
     * sigma, or if we are accumulating a substrate.  In either case,
     * we want to accumulate the zero length layer
     */
    /* if (dz == 0) continue; */

    /* Save the layer */
    assert(newi < n);
    d[newi] = dz;
    if (i == n) {
      /* printf("contract: adding final sld at %d\n",newi); */
      /* Last layer uses surface values */
      rho[newi] = rho[n-1];
      irho[newi] = irho[n-1];
      /* No interface for final layer */
    } else {
      /* Middle layers uses average values */
      rho[newi] = rhoarea / dz;
      irho[newi] = irhoarea / dz;
      sigma[newi] = sigma[i-1];
    } /* First layer uses substrate values */
    newi++;
  }

  return newi;
}

extern "C"
int
contract_mag(int n, double d[], double sigma[],
             double rho[], double irho[],
             double rhoM[], double thetaM[],
             double dA)
{
  double dz;
  double rholo, rhohi, rhoarea;
  double irholo, irhohi, irhoarea;
  double mparalo, mparahi, rhoMpara, rhoMpara_area;
  double mperplo, mperphi, rhoMperp, rhoMperp_area;
  double mean_rhoMpara, mean_rhoMperp, mean_rhoM, rhoM_sign;
  double thetaM_area, thetaM_phase_offset;
  double thetaM_radians, thetaM_from_mean;
  int i, newi, m;
  m = n - 1; /* last middle layer */
  i=newi=1; /* Skip the substrate */
  while (i < m) {

    /* Get ready for the next layer */
    /* Accumulation of the first row happens in the inner loop */
    dz = 0;
    rhoarea = irhoarea = rhoMpara_area = rhoMperp_area = thetaM_area = 0.0;
    rholo = rhohi = rho[i];
    irholo = irhohi = irho[i];

    /*
     * averaged thetaM is from atan2 (when rhoM is nonzero)
    * which returns values in the range -180 to 180,
    * so we keep track of the phase offset of the input
    * to match it afterward
    */
    thetaM_phase_offset = floor((thetaM[i] + 180.) / 360.0);

    /* Pre-calculate projections */
    rhoM_sign = copysign(1.0, rhoM[i]);
    thetaM_radians = thetaM[i] * M_PI / 180.0;
    rhoMpara = rhoM[i] * cos(thetaM_radians);
    rhoMperp = rhoM[i] * sin(thetaM_radians);
    /*
     * Note that mpara indicates M when theta_M = 0,
     * and mperp indicates M when theta_M = 90,
     * but in most cases M is parallel to H when theta_M = 270
     * and Aguide = 270
     */
    mparalo = mparahi = rhoMpara;
    mperplo = mperphi = rhoMperp;

    /* Accumulate slices into layer */
    while (i < m) {
      assert(i < m);
      /* Accumulate next slice */
      dz += d[i];
      rhoarea+=d[i]*rho[i];
      irhoarea+=d[i]*irho[i];
      /* thetaM_area is only used if rhoM is zero */
      thetaM_area += d[i] * thetaM[i];
      /* Use pre-calculated next values */
      rhoMpara_area += rhoMpara * d[i];
      rhoMperp_area += rhoMperp * d[i];

      /* If no more slices or sigma != 0, break immediately */
      if (++i == n || sigma[i-1] != 0.) break;

      /* If next slice exceeds limit then break */
      if (rho[i] < rholo) rholo = rho[i];
      if (rho[i] > rhohi) rhohi = rho[i];
      if ((rhohi-rholo)*(dz+d[i]) > dA) break;

      if (irho[i] < irholo) irholo = irho[i];
      if (irho[i] > irhohi) irhohi = irho[i];
      if ((irhohi-irholo)*(dz+d[i]) > dA) break;

      /* Pre-calculate projections of next layer */
      thetaM_radians = thetaM[i] * M_PI / 180.0;
      rhoMpara = rhoM[i] * cos(thetaM_radians);
      rhoMperp = rhoM[i] * sin(thetaM_radians);

      /* If next slice is wrapped in phase, break */
      if (floor((thetaM[i] + 180.0) / 360.0) != thetaM_phase_offset) break;

      /* If next slice has different sign for rhoM, break */
      if (copysign(1.0, rhoM[i]) != rhoM_sign) break;

      if (rhoMpara < mparalo) mparalo = rhoMpara;
      if (rhoMpara > mparahi) mparahi = rhoMpara;
      if ((mparahi-mparalo)*(dz+d[i]) > dA) break;

      if (rhoMperp < mperplo) mperplo = rhoMperp;
      if (rhoMperp > mperphi) mperphi = rhoMperp;
      if ((mperphi-mperplo)*(dz+d[i]) > dA) break;
    }

    /* Save the layer */
    assert(newi < m);
    d[newi] = dz;
    if (dz == 0) {
      /* Last layer uses surface values */
      rho[newi] = rho[i-1];
      irho[newi] = irho[i-1];
      rhoM[newi] = rhoM[i-1];
      thetaM[newi] = thetaM[i-1];
      /* No interface for final layer */
    } else {
      /* Middle layers uses average values */
      rho[newi] = rhoarea / dz;
      irho[newi] = irhoarea / dz;
      mean_rhoMpara = rhoMpara_area / dz;
      mean_rhoMperp = rhoMperp_area / dz;
      mean_rhoM = sqrt((mean_rhoMpara * mean_rhoMpara) + (mean_rhoMperp * mean_rhoMperp)) * rhoM_sign;
      if (mean_rhoM == 0.0) {
        /* If rhoM is zero, then thetaM is meaningless: use plain average */
        thetaM[newi] = thetaM_area / dz;
      } else {
        /* Otherwise, calculate the mean thetaM
         * invert the sign of components if rhoM is negative, before atan2 */
        thetaM_from_mean = atan2(mean_rhoMperp * rhoM_sign, mean_rhoMpara * rhoM_sign) * 180.0 / M_PI;
        thetaM_from_mean += 360.0 * thetaM_phase_offset;
        thetaM[newi] = thetaM_from_mean;
      }
      rhoM[newi] = mean_rhoM;
    } /* First layer uses substrate values */
    sigma[newi] = sigma[i-1];
    newi++;
  }

  /* Save the last layer */
  /* Last layer uses surface values */
  rho[newi] = rho[n-1];
  irho[newi] = irho[n-1];
  rhoM[newi] = rhoM[n-1];
  thetaM[newi] = thetaM[n-1];
  /* No interface for final layer */
  newi++;

  return newi;
}

#else
#error "dynamic programming solution not yet implemented"
// A correct solution will have to incorporate forced breaks at sigma != 0,
// with a strong preference to merge layers the closer you are to the
// interface. The current solution does not do this.
static void
find_breaks(int n, double d[], double rho[], double irho[], double dA,
            double cost[], int nextbreak[])
{
  const double breakpenalty=dA;
  int i;

  std::cout << "n=" << n << std::endl;
  cost[n]=0.;
  nextbreak[n]=-1;
  for (i=n-1; i >= 0; i--) {
    double dz,rhoarea,irhoarea,bestcost;
    int j, best;
    if (i%50==0) std::cout << "i=" << i << std::endl;

    dz=rhoarea=irhoarea=0.;
    bestcost = cost[i+1];
    best=i+1;
    for (j=i; j<n; j++) {
      double rhoj, irhoj;
      double rhocost, irhocost, costj;
      int k;

      //      std::cout << "i=" << i << ", j=" << j << std::endl;
      /* Maintain running average */
      rhoarea += rho[j]*d[j];
      irhoarea += irho[j]*d[j];
      dz += d[j];

      if (j==n-1) {        /* substrate layer: use substrate parameters */
	if (i==0) break;    /* infinity penalty for only one layer */
	rhoj=rho[j]; irhoj=irho[j];
      } else if (i == 0) { /* vacuum layer: use vacuum parameters */
	rhoj=rho[0]; irhoj=irho[0];
      } else {             /* intermediate layer: use average parameters */
	rhoj=rhoarea/dz; irhoj=irhoarea/dz;
      }

      /* simple penalty --- may need rms penalty if sharp spikes need
       * to be preserved */
      rhocost = irhocost = 0.;
      for (k=i; k <= j; k++) {
	//	std::cout << "i=" << i << ", j=" << j << ", k=" << k << std::endl;
	rhocost += fabs(rho[k]-rhoj)/d[j];
	irhocost += fabs(irho[k]-irhoj)/d[j];
      }

      /* cost is monotonically increasing.
       * Proof:
       *   Rho chosen as average of all current slices.
       *   Any other rho increases cost of current slices.
       *   If next slice changes rho, cost for current slices
       *   slices increases.  If next slice j doesn't change rho,
       *   cost increases by |rho_j-rho|*d_j >= 0, so total cost
       *   increases.
       * Since cost is increasing, limit our search to this
       * j if we have already exceeded the allowed cost of a
       * slice.  This changes the algorithm from O(n^2) to O(km)
       * where k is the average cost per slice.
       */
      if (rhocost > dA || irhocost > dA) break;
      costj =  rhocost + irhocost + cost[j+1];
      if (costj < bestcost) {
	best = j+1;
	bestcost = costj;
      }
    }

    cost[i] = bestcost + breakpenalty;
    nextbreak[i] = best;
  }
}

int merge_breaks(double d[], double rho[], double irho[], int nextbreak[])
{
  int i, next, newi;
  newi = 0;
  i = 0;
  next = nextbreak[0];
  while (next > 0) {
    double dz,rhoarea,irhoarea,bestcost;
    int j, best;

    /* Compute average */
    dz=rhoarea=irhoarea=0.;
    while (i < next) {
      rhoarea += rho[i]*d[i];
      irhoarea += irho[i]*d[i];
      dz += d[i];
      i++;
    }
    next = nextbreak[i];

    d[newi]=dz;
    if (next < 0) {        /* substrate layer: use substrate parameters */
      rho[newi]=rho[i-1];
      irho[newi]=irho[i-1];
    } else if (newi==0) { /* vacuum layer: use vacuum parameters */
      /* NOP since newi==0: rho[newi] = rho[0]; */
      /* NOP since newi==0: irho[newi] = irho[0]; */
    } else {               /* intermediate layer: use average parameters */
      rho[newi] = rhoarea/dz;
      irho[newi] = irhoarea/dz;
    }
    newi++;
  }
  return newi;
}

extern "C"
int
contract_by_area(int n, double d[], double sigma[],
                 double rho[], double irho[], double dA,
                 void *work)
{
   // Note: currently ignores sigma!
   find_breaks(n, d, rho, irho, dA, cost, breaks);
   int new_n = merge_breaks(d, rho, irho, breaks);
   return new_n;
}

#endif


// $Id: contract_profile.cc 2 2005-08-02 00:19:11Z pkienzle $
