// This program is public domain.

#include <iostream>
#include <fstream>
#include <complex>
#include <cstdlib>
#include "reflcalc.h"


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
      int I,L,STEP;

//    variables calculating S1, S3, COSH and SINH
      double E0;
      double EPA, EMA, COSB, SINB, LOGH;
      Cplx S1,S3,COSHS1,COSHS3,SINHS1,SINHS3;

//    completely unrolled matrices for B=A*B update
      Cplx DELTA,U1L,U3L;
      Cplx A11,A12,A13,A14,A21,A22,A23,A24;
      Cplx A31,A32,A33,A34,A41,A42,A43,A44;
      Cplx B11,B12,B13,B14,B21,B22,B23,B24;
      Cplx B31,B32,B33,B34,B41,B42,B43,B44;
      Cplx C1,C2,C3,C4;

//    variables for translating resulting B into a signal
      Cplx SCI,SS,CC;
      Cplx W11,W12,W21,W22,V11,V12,V21,V22;
      Cplx ZIP,ZIM,ZSP,ZSM,X,YPP,YMM,YPM,YMP;
      Cplx DETW;

//    constants
      const Cplx CR(1.0,0.0);
      const Cplx CI(0.0,1.0);
      const double PI4=12.566370614359172e-6;
      const double PI=3.1415926535897932284626;
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

/*
C Given
C   C+ = cosh(D*S1) + cosh(D*S3)
C   C- = cosh(D*S1) - cosh(D*S3)
C   S*+ = S1*sinh(D*S1) + S3*sinh(D*S3)
C   S*- = S1*sinh(D*S1) - S3*sinh(D*S3)
C   S/+ = sinh(D*S1)/S1 + sinh(D*S3)/S3
C   S/- = sinh(D*S1)/S1 - sinh(D*S3)/S3
C   pth = e^(j pi theta/180)
C   mth = e^(-j pi theta/180)
C   S1 = sqrt(Qc^2 + mQc^2 - j pi (8 mu/lambda) - (Q^2 + fronting Qc^2))/2
C   S3 = sqrt(Qc^2 - mQc^2 - j pi (8 mu/lambda) - (Q^2 + fronting Qc^2))/2
C   Qc^2 = 16 pi rho
C   H = max(abs(real(S1)),abs(real(S3)))
C   D, theta, mu, Qc^2 and mQc^2 are the parameters for layer L
C
C Construct the following matrix A(L)
C
C                /    C+  mthC-      S/+ mthS/- \
C               |                                |
C               |  pthC-     C+   pthS/-    S/+  |
C   A(L)= 0.5/H*|                                |
C               |     S*+ mthS*-     C+  mthC-   |
C               |                                |
C                \ pthS*-    S*+  pthC-     C+  /
C
C Multiply A by existing layers B=A(L)*A(L-1)*...A(1)*I
C Use the factor of 0.5 to keep diagonals as close to 1 as possible
C Use the factor of H to avoid cancellation errors in e.g., C-, which
C for large D would otherwise approach Inf-Inf.  These factors cancel
C later in the calculation when we divide by DETW.
*/

//     B = I
      B11=Cplx(1.0,0.0);
      B12=Cplx(0.0,0.0);
      B13=Cplx(0.0,0.0);
      B14=Cplx(0.0,0.0);
      B21=Cplx(0.0,0.0);
      B22=Cplx(1.0,0.0);
      B23=Cplx(0.0,0.0);
      B24=Cplx(0.0,0.0);
      B31=Cplx(0.0,0.0);
      B32=Cplx(0.0,0.0);
      B33=Cplx(1.0,0.0);
      B34=Cplx(0.0,0.0);
      B41=Cplx(0.0,0.0);
      B42=Cplx(0.0,0.0);
      B43=Cplx(0.0,0.0);
      B44=Cplx(1.0,0.0);

//    Changing the target KZ is equivalent to subtracting the fronting
//    medium SLD.
      if (IP > 0) {
        // IP = 1 specifies polarization of the incident beam I+
        E0 = KZ*KZ + PI4*(RHO[L]+RHOM[L]);
      } else {
        // IP = 0 specifies polarization of the incident beam I-
        E0 = KZ*KZ + PI4*(RHO[L]-RHOM[L]);
      }
      ZIP=CI*sqrt(E0-PI4*(RHO[L]+RHOM[L]) + CI*PI4*IRHO[L]);
      ZIM=CI*sqrt(E0-PI4*(RHO[L]-RHOM[L]) + CI*PI4*IRHO[L]);
//    Process the loop once for each interior layer, either from
//    front to back or back to front.
      for (I=1; I < N-1; I++) {
        L = L+STEP;
        S1 = sqrt(PI4*(RHO[L]+RHOM[L])-E0 - CI*PI4*IRHO[L]);
        S3 = sqrt(PI4*(RHO[L]-RHOM[L])-E0 - CI*PI4*IRHO[L]);
        U1L = U1[L];
        U3L = U3[L];

//    Factor out H=exp(max(abs(real([S1,S3])))*D(L)) from the matrix
        if (fabs(S1.real()) > fabs(S3.real()))
          LOGH = fabs(S1.real())*D[L];
        else
          LOGH = fabs(S3.real())*D[L];
LOGH=0;

//    Calculate 2*COSH/H and 2*SINH/H for D*S1
        X    = S1*D[L];
        EPA  = exp(X.real()-LOGH);
        EMA  = exp(-X.real()-LOGH);
        SINB = sin(X.imag());
        COSB = cos(X.imag());
        COSHS1 = (EPA+EMA)*COSB + CI*((EPA-EMA)*SINB);
        SINHS1 = (EPA-EMA)*COSB + CI*((EPA+EMA)*SINB);

//    Calculate 2*COSH/H and 2*SINH/H for D*S3
        X    = S3*D[L];
        EPA  = exp(X.real()-LOGH);
        EMA  = exp(-X.real()-LOGH);
        SINB = sin(X.imag());
        COSB = cos(X.imag());
        COSHS3 = (EPA+EMA)*COSB + CI*((EPA-EMA)*SINB);
        SINHS3 = (EPA-EMA)*COSB + CI*((EPA+EMA)*SINB);

//    Use DELTA instead of 2*DELTA because we are generating
//    2*cosh/H and 2*sinh/H rather than cosh/H and sinh/H
        DELTA = 0.5/(U3L - U1L);
        A11=DELTA*(U3L*COSHS1-U1L*COSHS3);
        A21=DELTA*U1L*U3L*(COSHS1-COSHS3);
        A31=DELTA*(U3L*SINHS1*S1-U1L*SINHS3*S3);
        A41=DELTA*U1L*U3L*(SINHS1*S1-SINHS3*S3);
        
        A12=-DELTA*(COSHS1-COSHS3);
        A22=-DELTA*(U1L*COSHS1-U3L*COSHS3);
        A32=-DELTA*(SINHS1*S1-SINHS3*S3);
        A42=-DELTA*(U1L*SINHS1*S1-U3L*SINHS3*S3);
        
        
        A13=DELTA*(U3L*SINHS1/S1-U1L*SINHS3/S3);
        A23=DELTA*U1L*U3L*(SINHS1/S1-SINHS3/S3);
        A33=A11;
        A43=A21;
        
        A14=-DELTA*(SINHS1/S1-SINHS3/S3);
        A24=-DELTA*(U1L*SINHS1/S1-U3L*SINHS3/S3);
        A34=A12;
        A44=A22;

#if 0
        std::cout << "cr4x A1:"<<A11<<" "<<A12<<" "<<A13<<" "<<A14<<std::endl;
        std::cout << "cr4x A2:"<<A21<<" "<<A22<<" "<<A23<<" "<<A24<<std::endl;
        std::cout << "cr4x A3:"<<A31<<" "<<A32<<" "<<A33<<" "<<A34<<std::endl;
        std::cout << "cr4x A4:"<<A41<<" "<<A42<<" "<<A43<<" "<<A44<<std::endl;
#endif

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
      }
//    Done computing B = A(N)*...*A(2)*A(1)*I
#if 0
        std::cout << "cr4x B1:"<<B11<<" "<<B12<<" "<<B13<<" "<<B14<<std::endl;
        std::cout << "cr4x B2:"<<B21<<" "<<B22<<" "<<B23<<" "<<B24<<std::endl;
        std::cout << "cr4x B3:"<<B31<<" "<<B32<<" "<<B33<<" "<<B34<<std::endl;
        std::cout << "cr4x B4:"<<B41<<" "<<B42<<" "<<B43<<" "<<B44<<std::endl;
#endif

//    Rotate polarization axis to lab frame (angle AGUIDE)
//    Note: reusing A instead of creating CST
      CC = cos(-AGUIDE/2.*PI/180.);
      SS = sin(-AGUIDE/2.*PI/180.);
      SCI = CI*CC*SS; CC *= CC; SS *= SS;
      A11 = CC*B11 + SS*B22 + SCI*(B12-B21);
      A12 = CC*B12 + SS*B21 + SCI*(B11-B22);
      A21 = CC*B21 + SS*B12 + SCI*(B22-B11);
      A22 = CC*B22 + SS*B11 + SCI*(B21-B12);
      A13 = CC*B13 + SS*B24 + SCI*(B14-B23);
      A14 = CC*B14 + SS*B23 + SCI*(B13-B24);
      A23 = CC*B23 + SS*B14 + SCI*(B24-B13);
      A24 = CC*B24 + SS*B13 + SCI*(B23-B14);
      A31 = CC*B31 + SS*B42 + SCI*(B32-B41);
      A32 = CC*B32 + SS*B41 + SCI*(B31-B42);
      A41 = CC*B41 + SS*B32 + SCI*(B42-B31);
      A42 = CC*B42 + SS*B31 + SCI*(B41-B32);
      A33 = CC*B33 + SS*B44 + SCI*(B34-B43);
      A34 = CC*B34 + SS*B43 + SCI*(B33-B44);
      A43 = CC*B43 + SS*B34 + SCI*(B44-B33);
      A44 = CC*B44 + SS*B33 + SCI*(B43-B34);

#if 0
        std::cout << "cr4x A1:"<<A11<<" "<<A12<<" "<<A13<<" "<<A14<<std::endl;
        std::cout << "cr4x A2:"<<A21<<" "<<A22<<" "<<A23<<" "<<A24<<std::endl;
        std::cout << "cr4x A3:"<<A31<<" "<<A32<<" "<<A33<<" "<<A34<<std::endl;
        std::cout << "cr4x A4:"<<A41<<" "<<A42<<" "<<A43<<" "<<A44<<std::endl;
#endif

//    Use corrected versions of X,Y,ZI, and ZS to account for effect
//    of incident and substrate media
//    Note: this does not take into account magnetic fronting/backing
//    media --- use gepore.f directly for a more complete solution
      L=L+STEP;
      ZSP=CI*sqrt(E0-PI4*(RHO[L]+RHOM[L]) + CI*PI4*IRHO[L]);
      ZSM=CI*sqrt(E0-PI4*(RHO[L]-RHOM[L]) + CI*PI4*IRHO[L]);
//    looking for ZIP and ZIM?  They have been moved to the top of the loop.      

      X=-1.;
      YPP=ZIP*ZSP;
      YMM=ZIM*ZSM;
      YPM=ZIP*ZSM;
      YMP=ZIM*ZSP;

//    W below is U and V is -V of printed versions

      V11=ZSP*A11+X*A31+YPP*A13-ZIP*A33;
      V12=ZSP*A12+X*A32+YMP*A14-ZIM*A34;
      V21=ZSM*A21+X*A41+YPM*A23-ZIP*A43;
      V22=ZSM*A22+X*A42+YMM*A24-ZIM*A44;

      W11=ZSP*A11+X*A31-YPP*A13+ZIP*A33;
      W12=ZSP*A12+X*A32-YMP*A14+ZIM*A34;
      W21=ZSM*A21+X*A41-YPM*A23+ZIP*A43;
      W22=ZSM*A22+X*A42-YMM*A24+ZIM*A44;

      DETW=W22*W11-W12*W21;

//    Calculate reflectivity coefficients specified by POLSTAT
      YA = (V21*W12-V11*W22)/DETW;
      YB = (V11*W21-V21*W11)/DETW;
      YC = (V22*W12-V12*W22)/DETW;
      YD = (V12*W21-V22*W11)/DETW;

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
  if (rhoM[0] == 0.0 && rhoM[layers-1] == 0.0) {
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
    //ip = 0; // minus polarization
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i < points; i++) {
      const int offset = layers*(rho_index != NULL?rho_index[i]:0);
      Cr4xa(layers,d,sigma,ip,rho+offset,irho+offset,rhoM,u1,u3,
            Aguide,KZ[i],Ra[i],Rb[i],dummy1,dummy2);
    }
    //ip = 1; // plus polarization
    ip = 0; // minus polarization
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
