C This program is public domain.

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
C EXPTH(N) is the magnetic scattering vector
C    EXPTH(k) = exp(1i P(k)), where P(k) is the angle in radians
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
C Revision 1.1  2005/08/02 00:18:24  pkienzle
C initial release
C
C 2005-02-17 Paul Kienzle
C * No need to precompute S
C * Support for absorption in substrate
C 2004-04-29 Paul Kienzle
C * Handle negative Q by reversing the loop
C * Only calculate single Q
C 2002-01-08 Paul Kienzle
C * Optimizations by precomputing layer parameter values
C 2001-03-26 Kevin O`Donovan
C * Converted to subroutine from GEPORE.f

      subroutine R4XA(N,D,RHO,MU,LAMBDA,P,EXPTH,AGUIDE,Q,YA,YB,YC,YD)
      implicit none

C     paramters
      integer N,I,L,STEP
      double complex EXPTH(1)
      double precision Q,D(1),P(1),RHO(1),MU(1),LAMBDA,AGUIDE
      double complex YA,YB,YC,YD
        
C     variables calculating S1, S3, COSH and SINH
      double precision QSQREL, PI2oLAMBDA
      double precision EPA, EMA, COSB, SINB, LOGH
      double complex S1,S3,COSHS1,COSHS3,SINHS1,SINHS3

C     completely unrolled matrices for B=A*B update
      double complex A11,A12,A13,A14,A21,A22,A23,A24
      double complex A31,A32,A33,A34,A41,A42,A43,A44
      double complex B11,B12,B13,B14,B21,B22,B23,B24
      double complex B31,B32,B33,B34,B41,B42,B43,B44
      double complex C1,C2,C3,C4

c     variables for translating resulting B into a signal
      double complex W11,W12,W21,W22,V11,V12,V21,V22
      double complex DETW
      double complex ZI,ZS,X,Y,SCI,SS,CC

c     constants
      double complex CR,CI
      double precision PI4,PI
      parameter (CI=(0.0,1.0),CR=(1.0,0.0))
      parameter (PI4=1.2566370614359172D1)
      parameter (PI=3.1415926535897932284626)
c     Check for Q near zero.  If Q < 0, reverse the indices
      IF (Q.LE.-1.D-10) THEN
         L=N
         STEP=-1
      ELSEIF (Q.GE.1.D-10) THEN
         L=1
         STEP=1
      ELSE
         YA = -1.
         YB = 0.
         YC = 0.
         YD = -1.
         RETURN
      ENDIF

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

C     B = I
      B11=(1.0,0.0)
      B12=(0.0,0.0)
      B13=(0.0,0.0)
      B14=(0.0,0.0)
      B21=(0.0,0.0)
      B22=(1.0,0.0)
      B23=(0.0,0.0)
      B24=(0.0,0.0)
      B31=(0.0,0.0)
      B32=(0.0,0.0)
      B33=(1.0,0.0)
      B34=(0.0,0.0)
      B41=(0.0,0.0)
      B42=(0.0,0.0)
      B43=(0.0,0.0)
      B44=(1.0,0.0)

C     Changing the target Q is equivalent to subtracting the fronting
C     medium SLD.  Units are Nb so multiply by 16pi. We want sqrt(S-Q^2)/2
C     so divide by 4.
      QSQREL = 0.25*Q*Q + PI4*RHO(L)
      PI2oLAMBDA = PI4/(2.*LAMBDA)
C     Process the loop once for each interior layer, either from
C     front to back or back to front.
      DO 300 I=2,N-1
        L = L+STEP
        S1 = CDSQRT(PI4*(RHO(L)+P(L))-QSQREL - CI*PI2oLAMBDA*MU(L))
        S3 = CDSQRT(PI4*(RHO(L)-P(L))-QSQREL - CI*PI2oLAMBDA*MU(L))

C     Factor out H=exp(max(abs(real([S1,S3])))*D(L)) from the matrix
        IF (ABS(DREAL(S1)).GT.ABS(DREAL(S3))) THEN
          LOGH = ABS(DREAL(S1))*D(L)
        ELSE
          LOGH = ABS(DREAL(S3))*D(L)
        ENDIF

C     Calculate 2*COSH/H and 2*SINH/H for D*S1
        X    = S1*D(L)
        EPA  = EXP(DREAL(X)-LOGH)
        EMA  = EXP(-DREAL(X)-LOGH)
        SINB = SIN(DIMAG(X))
        COSB = COS(DIMAG(X))
        COSHS1 = (EPA+EMA)*COSB + CI*((EPA-EMA)*SINB)
        SINHS1 = (EPA-EMA)*COSB + CI*((EPA+EMA)*SINB)

C     Calculate 2*COSH/H and 2*SINH/H for D*S3
        X    = S3*D(L)
        EPA  = EXP(DREAL(X)-LOGH)
        EMA  = EXP(-DREAL(X)-LOGH)
        SINB = SIN(DIMAG(X))
        COSB = COS(DIMAG(X))
        COSHS3 = (EPA+EMA)*COSB + CI*((EPA-EMA)*SINB)
        SINHS3 = (EPA-EMA)*COSB + CI*((EPA+EMA)*SINB)

C     Generate A using a factor of 0.25 since we are using
C     2*cosh/H and 2*sinh/H rather than cosh/H and sinh/H
        A11=0.25*(COSHS1+COSHS3)
        A21=0.25*(COSHS1-COSHS3)
        A31=0.25*(SINHS1*S1+SINHS3*S3)
        A41=0.25*(SINHS1*S1-SINHS3*S3)
        A13=0.25*(SINHS1/S1+SINHS3/S3)
        A23=0.25*(SINHS1/S1-SINHS3/S3)
        A32=A41*CONJG(EXPTH(L))
        A14=A23*CONJG(EXPTH(L))
        A12=A21*CONJG(EXPTH(L))
        A41=A41*EXPTH(L)
        A23=A23*EXPTH(L)
        A21=A21*EXPTH(L)
        A43=A21
        A34=A12
        A22=A11
        A33=A11
        A44=A11
        A24=A13
        A42=A31
        
C     Matrix update B=A*B
        C1=A11*B11+A12*B21+A13*B31+A14*B41
        C2=A21*B11+A22*B21+A23*B31+A24*B41
        C3=A31*B11+A32*B21+A33*B31+A34*B41
        C4=A41*B11+A42*B21+A43*B31+A44*B41
        B11=C1
        B21=C2
        B31=C3
        B41=C4
        
        C1=A11*B12+A12*B22+A13*B32+A14*B42
        C2=A21*B12+A22*B22+A23*B32+A24*B42
        C3=A31*B12+A32*B22+A33*B32+A34*B42
        C4=A41*B12+A42*B22+A43*B32+A44*B42
        B12=C1
        B22=C2
        B32=C3
        B42=C4
        
        C1=A11*B13+A12*B23+A13*B33+A14*B43
        C2=A21*B13+A22*B23+A23*B33+A24*B43
        C3=A31*B13+A32*B23+A33*B33+A34*B43
        C4=A41*B13+A42*B23+A43*B33+A44*B43
        B13=C1
        B23=C2
        B33=C3
        B43=C4
        
        C1=A11*B14+A12*B24+A13*B34+A14*B44
        C2=A21*B14+A22*B24+A23*B34+A24*B44
        C3=A31*B14+A32*B24+A33*B34+A34*B44
        C4=A41*B14+A42*B24+A43*B34+A44*B44
        B14=C1
        B24=C2
        B34=C3
        B44=C4

 300  CONTINUE
C     Done computing B = A(N)*...*A(2)*A(1)*I

C     Rotate polarization axis to lab frame (angle AGUIDE)
C     Note: reusing A instead of creating CST
      CC = COS(-AGUIDE/2.*PI/180.)**2
      SS = SIN(-AGUIDE/2.*PI/180.)**2
      SCI = CI*COS(-AGUIDE/2.*PI/180.)*SIN(-AGUIDE/2*PI/180.)
      A11 = CC*B11 + SS*B22 + SCI*(B12-B21)
      A12 = CC*B12 + SS*B21 + SCI*(B11-B22)
      A21 = CC*B21 + SS*B12 + SCI*(B22-B11)
      A22 = CC*B22 + SS*B11 + SCI*(B21-B12)
      A13 = CC*B13 + SS*B24 + SCI*(B14-B23)
      A14 = CC*B14 + SS*B23 + SCI*(B13-B24)
      A23 = CC*B23 + SS*B14 + SCI*(B24-B13)
      A24 = CC*B24 + SS*B13 + SCI*(B23-B14)
      A31 = CC*B31 + SS*B42 + SCI*(B32-B41)
      A32 = CC*B32 + SS*B41 + SCI*(B31-B42)
      A41 = CC*B41 + SS*B32 + SCI*(B42-B31)
      A42 = CC*B42 + SS*B31 + SCI*(B41-B32)
      A33 = CC*B33 + SS*B44 + SCI*(B34-B43)
      A34 = CC*B34 + SS*B43 + SCI*(B33-B44)
      A43 = CC*B43 + SS*B34 + SCI*(B44-B33)
      A44 = CC*B44 + SS*B33 + SCI*(B43-B34)
      
C     Use corrected versions of X,Y,ZI, and ZS to account for effect
C     of incident and substrate media
C     Note: this does not take into account magnetic fronting/backing
C     media --- use gepore.f directly for a more complete solution
      L=L+STEP
      ZS=CI*CDSQRT(QSQREL-PI4*RHO(L) + CI*PI2oLAMBDA*MU(L))
      ZI=CI*DABS(0.5*Q)
 
      X=-1.
      Y=ZI*ZS
      
C     W below is U and V is -V of printed versions
      
      V11=ZS*A11+X*A31+Y*A13-ZI*A33
      V12=ZS*A12+X*A32+Y*A14-ZI*A34
      V21=ZS*A21+X*A41+Y*A23-ZI*A43
      V22=ZS*A22+X*A42+Y*A24-ZI*A44
      
      W11=ZS*A11+X*A31-Y*A13+ZI*A33
      W12=ZS*A12+X*A32-Y*A14+ZI*A34
      W21=ZS*A21+X*A41-Y*A23+ZI*A43
      W22=ZS*A22+X*A42-Y*A24+ZI*A44
      
      DETW=W22*W11-W12*W21
     
C     Calculate reflectivity coefficients specified by POLSTAT
      YA = (V21*W12-V11*W22)/DETW
      YB = (V11*W21-V21*W11)/DETW
      YC = (V22*W12-V12*W22)/DETW
      YD = (V12*W21-V22*W11)/DETW

      return
      END
