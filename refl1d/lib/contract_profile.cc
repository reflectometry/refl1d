// This program is public domain.

#include <iostream>

#define GREEDY

#include <math.h>
#include <assert.h>

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

    if (dz == 0) continue;

    /* Save the layer */
    assert(newi < n);
    d[newi] = dz;
    if (i == n) {
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
contract_mag(int n, double d[],
             double rho[], double irho[],
             double rhoM[], double thetaM[],
             double dA)
{
  double dz, weighted_dz, weight;
  double rholo, rhohi, rhoarea;
  double irholo, irhohi, irhoarea;
  double maglo, maghi, mag, rhoMarea, thetaMarea;
  int i, newi;
  i=newi=1; /* Skip the substrate */
  while (i < n) {

    /* Get ready for the next layer */
    /* Accumulation of the first row happens in the inner loop */
    dz = weighted_dz = 0;
    rhoarea = irhoarea = rhoMarea = thetaMarea = 0.;
    rholo=rhohi=rho[i];
    irholo=irhohi=irho[i];
    maglo=maghi=rhoM[i]*cos(thetaM[i]*M_PI/180.);

    /* Accumulate slices into layer */
    for (;;) {
      assert(i < n);
      /* Accumulate next slice */
      dz += d[i];
      rhoarea+=d[i]*rho[i];
      irhoarea+=d[i]*irho[i];

      /* Weight the magnetic signal by the in-plane contribution
       * when accumulating rhoM and thetaM. */
      weight = cos(thetaM[i]*M_PI/180.);
      mag = rhoM[i]*weight;
      rhoMarea += d[i]*rhoM[i]*weight;
      thetaMarea += d[i]*thetaM[i]*weight;
      weighted_dz += d[i]*weight;

      /* If no more slices break immediately */
      if (++i == n) break;

      /* If next slice exceeds limit then break */
      if (rho[i] < rholo) rholo = rho[i];
      if (rho[i] > rhohi) rhohi = rho[i];
      if ((rhohi-rholo)*(dz+d[i]) > dA) break;

      if (irho[i] < irholo) irholo = irho[i];
      if (irho[i] > irhohi) irhohi = irho[i];
      if ((irhohi-irholo)*(dz+d[i]) > dA) break;

      if (mag < maglo) maglo = mag;
      if (mag > maghi) maghi = mag;
      if ((maghi-maglo)*(dz+d[i]) > dA) break;
    }

    if (dz == 0) continue;

    /* Save the layer */
    assert(newi < n);
    d[newi] = dz;
    if (i == n) {
      /* Last layer uses surface values */
      rho[newi] = rho[n-1];
      irho[newi] = irho[n-1];
      rhoM[newi] = rhoM[n-1];
      thetaM[newi] = thetaM[n-1];
      /* No interface for final layer */
    } else {
      /* Middle layers uses average values */
      rho[newi] = rhoarea / dz;
      irho[newi] = irhoarea / dz;
      rhoM[newi] = rhoMarea / weighted_dz;
      thetaM[newi] = thetaMarea / weighted_dz;
    } /* First layer uses substrate values */
    newi++;
  }

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
