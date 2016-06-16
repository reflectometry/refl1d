// This program is public domain.

#include <iostream>
#include <fstream>
#include <complex>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "reflcalc.h"

#define MINIMAL_RHO_M 1e-2  // in units of 1e-6/A^2
const double EPS = std::numeric_limits<double>::epsilon();

extern "C" void
Cr4xa(const int &N, const double D[], const double SIGMA[],
      const int &IP,
      const double RHO[], const double IRHO[],
      const double RHOM[], const Cplx U1[], const Cplx U3[],
      const double &AGUIDE, const double &KZ,
      Cplx &YA, Cplx &YB, Cplx &YC, Cplx &YD)

{
/*
C Modification of C.F. Majrkzak`s progam gepore.f for calculating
C reflectivities of four polarization states of neutron reflectivity data.

c ****************************************************************
c
c Program "gepore.f" (GEneral POlarized REflectivity) calculates the
c spin-dependent neutron reflectivities (and transmissions) for
c model potentials, or scattering length density profiles, assuming
c the specular condition.
c
c In the present version, both nuclear and magnetic, real scattering
c length densities can be input, whereas imaginary components of the
c nuclear potential cannot.  Also, magnetic and nuclear incident, or
c "fronting", and substrate, or "backing", media can be included.  A
c description of the input parameters is given below:
c
c It must be noted that in the continuum reflectivity calculation
c performed by this program, Maxwell`s equations apply, specifically
c the requirement that the component of the magnetic induction, B,
c normal to a boundary surface be continuous.  Neither the program
c nor the wave equation itself automatically insure that this is so:
c this condition must be satisfied by appropriate selection of the
c magnetic field direction in the incident and substrate media.

C R4XA(Q,N,D,S,P,EXPTH,A,B,C,D) returns the complex amplitude r

C Q is the value to calculate in inverse Angstroms
C N is the number of layers
C D(N) is the layer depth in angstroms
C S(N) is the complex scattering length density in number density units
C    S(k) = RHO(k) - 0.5i MU(k)/LAMBDA
C P(N) is the magnetic scattering length density in number density units
C U1[N] is U(1) from gepore.f
C U3[N] is U(3) from gepore.f
C
C A,B,C,D is the result for the ++, -+, +- and -- cross sections
C
C Notes:
C
C 1. If Q is negative then the beam is assumed to come in from the
C bottom of the sample, and all the layers are reversed.
C
C 2. The fronting and backing materials are assumed to be semi-infinite,
C so depth is ignored for the first and last layer.
C
C 3. Absorption is ignored for the fronting material, or the backing
C material for negative Q.  For negative Q, the beam is coming in
C through the side of the substrate, and you will need to multiply
C by a substrate absorption factor depending on the path length through
C the substrate.  For neutron reflectivity, this is approximately
C constant for the angles we need to consider.
C
C 4. Magnetic scattering is ignored for the fronting and backing.
C
C 5. This subroutine does not deal with any component of sample moment
C that may lie out of the plane of the film.  Such a perpendicular
C component will cause a neutron presession, therefore an additional
C spin flip term.  If reflectivity data from a sample with an
C out-of-plane moment is modeled using this subroutine, one will
C obtain erroneous results, since all of the spin flip scattering
C will be attributed to in-plane moments perpendicular to the neutron.


C $Log$
C Modification 2014/11/25 Brian Maranville
C specifying polarization state of incoming beam
C to allow for Felcher effect
C 
C Revision 1.1  2005/08/02 00:18:24  pkienzle
C initial release
C
C 2005-02-17 Paul Kienzle
C * No need to precompute S
C * Support for absorption in substrate
C 2004-04-29 Paul Kienzle
C * Handle negative KZ by reversing the loop
C * Only calculate single KZ
C 2002-01-08 Paul Kienzle
C * Optimizations by precomputing layer parameter values
C 2001-03-26 Kevin O`Donovan
C * Converted to subroutine from GEPORE.f
*/

//     paramters
      int I,L,LP,STEP;

//    variables calculating S1, S3, and exponents
      double E0;
      Cplx S1L,S3L,S1LP,S3LP,ES1L,ES3L,ENS1L,ENS3L,ES1LP,ES3LP,ENS1LP,ENS3LP;
      Cplx FS1S1, FS3S1, FS1S3, FS3S3;

//    completely unrolled matrices for B=A*B update
      Cplx DELTA,GL,GLP,BL,BLP, SSWAP;
      Cplx DBG, DBB, DGB, DGG; // deltas
      Cplx Z;
      Cplx A11,A12,A13,A14,A21,A22,A23,A24;
      Cplx A31,A32,A33,A34,A41,A42,A43,A44;
      Cplx B11,B12,B13,B14,B21,B22,B23,B24;
      Cplx B31,B32,B33,B34,B41,B42,B43,B44;
      Cplx C1,C2,C3,C4;
      //bool subcrit_plus = false, subcrit_minus = false; 

//    variables for translating resulting B into a signal
      Cplx DETW;

//    constants
      const Cplx CR(1.0,0.0);
      const Cplx CI(0.0,1.0);
      const double PI4=12.566370614359172e-6;
//    Check for KZ near zero.  If KZ < 0, reverse the indices
      if (KZ<=-1.e-10) {
         L=N-1;
         STEP=-1;
      } else if (KZ>=1.e-10) {
         L=0;
         STEP=1;
      } else {
         YA = -1.;
         YB = 0.;
         YC = 0.;
         YD = -1.;
         return;
      }


//    Changing the target KZ is equivalent to subtracting the fronting
//    medium SLD.
      if (IP > 0) {
        // IP = 1 specifies polarization of the incident beam I+
        E0 = KZ*KZ + PI4*(RHO[L]+RHOM[L]);
      } else {
        // IP = -1 specifies polarization of the incident beam I-
        E0 = KZ*KZ + PI4*(RHO[L]-RHOM[L]);
      }
      
      Z = 0.0;
      if (N>1) {
        // chi in layer 1
        LP = L + STEP;
        // Branch selection:  the -sqrt below for S1 and S3 will be 
        //     +Imag for KZ > Kcrit, 
        //     -Real for KZ < Kcrit
        // which covers the S1, S3 waves allowed by the boundary conditions in the 
        // fronting and backing medium:
        // either traveling forward (+Imag) or decaying exponentially forward (-Real).
        // The decaying exponential only occurs for the transmitted forward wave in the backing: 
        // the root +iKz is automatically chosen for the incident wave in the fronting.
        // 
        // In the fronting, the -S1 and -S3 waves are either traveling waves backward (-Imag) 
        // or decaying along the -z reflection direction (-Real) * (-z) = (+Real*z).
        // NB: This decaying reflection only occurs when the reflected wave is below Kcrit
        // while the incident wave is above Kcrit, so it only happens for spin-flip from 
        // minus to plus (lower to higher potential energy) and the observed R-+ will 
        // actually be zero at large distances from the interface.
        // 
        // In the backing, the -S1 and -S3 waves are explicitly set to be zero amplitude
        // by the boundary conditions (neutrons only incident in the fronting medium - no 
        // source of neutrons below).
        // 
        S1L = -sqrt(Cplx(PI4*(RHO[L]+RHOM[L])-E0, -PI4*(fabs(IRHO[L])+EPS)));
        S3L = -sqrt(Cplx(PI4*(RHO[L]-RHOM[L])-E0, -PI4*(fabs(IRHO[L])+EPS)));
        S1LP = -sqrt(Cplx(PI4*(RHO[LP]+RHOM[LP])-E0, -PI4*(fabs(IRHO[LP])+EPS)));
        S3LP = -sqrt(Cplx(PI4*(RHO[LP]-RHOM[LP])-E0, -PI4*(fabs(IRHO[LP])+EPS)));

        if (abs(U1[L]) <= 1.0) {
            // then Bz >= 0
            // BL and GL are zero in the fronting.
        } else {
            // then Bz < 0: flip!
            // This is probably impossible, since Bz defines the +z direction
            // in the fronting medium, but just in case...
            SSWAP = S1L;
            S1L = S3L;
            S3L = SSWAP; // swap S3 and S1
        }
        
        if (abs(U1[LP]) <= 1.0) {
            // then Bz >= 0
            BLP = U1[LP];
            GLP = 1.0/U3[LP];
        } else {
            // then Bz < 0: flip!
            BLP = U3[LP];
            GLP = 1.0/U1[LP];
            SSWAP = S1LP;
            S1LP = S3LP;
            S3LP = SSWAP; // swap S3 and S1
        }
        
        DELTA = 0.5*CR / (1.0 - (BLP*GLP));
        
        FS1S1 = S1L/S1LP;
        FS1S3 = S1L/S3LP;
        FS3S1 = S3L/S1LP;
        FS3S3 = S3L/S3LP;
         
        B11 = DELTA *   1.0 * (1.0 + FS1S1);
        B12 = DELTA *   1.0 * (1.0 - FS1S1);
        B13 = DELTA *  -GLP * (1.0 + FS3S1);
        B14 = DELTA *  -GLP * (1.0 - FS3S1);
        
        B21 = DELTA *   1.0 * (1.0 - FS1S1);
        B22 = DELTA *   1.0 * (1.0 + FS1S1);
        B23 = DELTA *  -GLP * (1.0 - FS3S1);
        B24 = DELTA *  -GLP * (1.0 + FS3S1);
        
        B31 = DELTA *  -BLP * (1.0 + FS1S3);
        B32 = DELTA *  -BLP * (1.0 - FS1S3);
        B33 = DELTA *   1.0 * (1.0 + FS3S3);
        B34 = DELTA *   1.0 * (1.0 - FS3S3);
        
        B41 = DELTA *  -BLP * (1.0 - FS1S3);
        B42 = DELTA *  -BLP * (1.0 + FS1S3);
        B43 = DELTA *   1.0 * (1.0 - FS3S3);
        B44 = DELTA *   1.0 * (1.0 + FS3S3);
        
        Z += D[LP];
        L = LP;
      }
      
//    Process the loop once for each interior layer, either from
//    front to back or back to front.
      for (I=1; I < N-1; I++) {
        LP = L + STEP;
        S1L = S1LP; // copy from the layer before
        S3L = S3LP; //
        GL = GLP;
        BL = BLP;
        S1LP = -sqrt(Cplx(PI4*(RHO[LP]+RHOM[LP])-E0, -PI4*(fabs(IRHO[L])+EPS)));
        S3LP = -sqrt(Cplx(PI4*(RHO[LP]-RHOM[LP])-E0, -PI4*(fabs(IRHO[LP])+EPS)));
        if (abs(U1[LP]) <= 1.0) {
            // then Bz >= 0
            BLP = U1[LP];
            GLP = 1.0/U3[LP];
        } else {
            // then Bz < 0: flip!
            BLP = U3[LP];
            GLP = 1.0/U1[LP];
            SSWAP = S1LP;
            S1LP = S3LP;
            S3LP = SSWAP; // swap S3 and S1
        }

        DELTA = 0.5*CR / (1.0 - (BLP*GLP));
        DBB = (BL - BLP) * DELTA; // multiply by delta here?
        DBG = (1.0 - BL*GLP) * DELTA;
        DGB = (1.0 - GL*BLP) * DELTA;
        DGG = (GL - GLP) * DELTA;

        ES1L = exp(S1L*Z);
        ENS1L = CR / ES1L;
        ES1LP = exp(S1LP*Z);
        ENS1LP = CR / ES1LP;
        ES3L = exp(S3L*Z);
        ENS3L = CR / ES3L;
        ES3LP = exp(S3LP*Z);
        ENS3LP = CR / ES3LP;
        
        FS1S1 = S1L/S1LP;
        FS1S3 = S1L/S3LP;
        FS3S1 = S3L/S1LP;
        FS3S3 = S3L/S3LP;

        A11 = A22 = DBG * (1.0 + FS1S1);
        A11 *= ES1L * ENS1LP;
        A22 *= ENS1L * ES1LP;
        A12 = A21 = DBG * (1.0 - FS1S1);
        A12 *= ENS1L * ENS1LP;
        A21 *= ES1L  * ES1LP;
        A13 = A24 = DGG * (1.0 + FS3S1);
        A13 *= ES3L  * ENS1LP;
        A24 *= ENS3L * ES1LP;
        A14 = A23 = DGG * (1.0 - FS3S1);
        A14 *= ENS3L * ENS1LP;
        A23 *= ES3L  * ES1LP;
        
        A31 = A42 = DBB * (1.0 + FS1S3);
        A31 *= ES1L * ENS3LP;
        A42 *= ENS1L * ES3LP;
        A32 = A41 = DBB * (1.0 - FS1S3);
        A32 *= ENS1L * ENS3LP;
        A41 *= ES1L  * ES3LP;
        A33 = A44 = DGB * (1.0 + FS3S3);
        A33 *= ES3L * ENS3LP;
        A44 *= ENS3L * ES3LP;
        A34 = A43 = DGB * (1.0 - FS3S3);
        A34 *= ENS3L * ENS3LP;
        A43 *= ES3L * ES3LP;


//    Matrix update B=A*B
        C1=A11*B11+A12*B21+A13*B31+A14*B41;
        C2=A21*B11+A22*B21+A23*B31+A24*B41;
        C3=A31*B11+A32*B21+A33*B31+A34*B41;
        C4=A41*B11+A42*B21+A43*B31+A44*B41;
        B11=C1;
        B21=C2;
        B31=C3;
        B41=C4;

        C1=A11*B12+A12*B22+A13*B32+A14*B42;
        C2=A21*B12+A22*B22+A23*B32+A24*B42;
        C3=A31*B12+A32*B22+A33*B32+A34*B42;
        C4=A41*B12+A42*B22+A43*B32+A44*B42;
        B12=C1;
        B22=C2;
        B32=C3;
        B42=C4;

        C1=A11*B13+A12*B23+A13*B33+A14*B43;
        C2=A21*B13+A22*B23+A23*B33+A24*B43;
        C3=A31*B13+A32*B23+A33*B33+A34*B43;
        C4=A41*B13+A42*B23+A43*B33+A44*B43;
        B13=C1;
        B23=C2;
        B33=C3;
        B43=C4;

        C1=A11*B14+A12*B24+A13*B34+A14*B44;
        C2=A21*B14+A22*B24+A23*B34+A24*B44;
        C3=A31*B14+A32*B24+A33*B34+A34*B44;
        C4=A41*B14+A42*B24+A43*B34+A44*B44;
        B14=C1;
        B24=C2;
        B34=C3;
        B44=C4;

        Z += D[LP];
        L = LP;
      }
//    Done computing B = A(N)*...*A(2)*A(1)*I

      DETW=(B44*B22-B24*B42);


//    Calculate reflectivity coefficients specified by POLSTAT
      YA = (B24*B41-B21*B44)/DETW; // ++ 
      YB = (B21*B42-B41*B22)/DETW; // +-
      YC = (B24*B43-B23*B44)/DETW; // -+ 
      YD = (B23*B42-B43*B22)/DETW; // --

}

extern "C" void
magnetic_amplitude(const int layers,
                      const double d[], const double sigma[],
                      const double rho[], const double irho[],
                      const double rhoM[], const Cplx u1[], const Cplx u3[],
                      const double Aguide,
                      const int points, const double KZ[], const int rho_index[],
                      Cplx Ra[], Cplx Rb[], Cplx Rc[], Cplx Rd[])
{
  Cplx dummy1,dummy2;
  int ip;
  if (fabs(rhoM[0]) <= MINIMAL_RHO_M && fabs(rhoM[layers-1]) <= MINIMAL_RHO_M) {
    ip = 1; // calculations for I+ and I- are the same in the fronting and backing.
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i < points; i++) {
      const int offset = layers*(rho_index != NULL?rho_index[i]:0);
      Cr4xa(layers,d,sigma,ip,rho+offset,irho+offset,rhoM,u1,u3,
            Aguide,KZ[i],Ra[i],Rb[i],Rc[i],Rd[i]);
    }
  } else {
    ip = 1; // plus polarization
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i < points; i++) {
      const int offset = layers*(rho_index != NULL?rho_index[i]:0);
      Cr4xa(layers,d,sigma,ip,rho+offset,irho+offset,rhoM,u1,u3,
            Aguide,KZ[i],Ra[i],Rb[i],dummy1,dummy2);
    }
    ip = -1; // minus polarization
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i < points; i++) {
      const int offset = layers*(rho_index != NULL?rho_index[i]:0);
      Cr4xa(layers,d,sigma,ip,rho+offset,irho+offset,rhoM,u1,u3,
            Aguide,KZ[i],dummy1,dummy2,Rc[i],Rd[i]);
    }
  }
}


// $Id: magnetic.cc 236 2007-05-30 17:15:57Z pkienzle $
