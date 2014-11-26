	PROGRAM GEPORE
c ****************************************************************
c
c Program "gepore.f" (GEneral POlarized REflectivity) calculates the
c spin-dependent neutron reflectivities (and transmissions) for
c model potentials, or scattering length density profiles, assuming
c the specular condition.
c 
c In the present version, both nuclear and magnetic, real scattering
c length densities can be input, whereas imaginary components of the
c nuclear potential cannot. Also, magnetic and nuclear incident, or
c "fronting", and substrate, or "backing", media can be included. A
c description of the input parameters is given below:
c
c 
c 	NL = # of distict layers or "slabs" of uniform scattering
c 	      length density (SLD)
c 
c 	NC = # of "unit cell" repeats
c
c 	QS = first wavevector transfer at which reflectivities and
c 	     transmissions are calculated (Angstroms -1)
c
c 	DQ = increment in Q (A-1)
c
c 	NQ = # of Q-values at which reflectivities and transmissions
c 	     are calculated
c
c 	EPS = angle, in radians, between the laboratory guide field
c 	      or quantization axis and the sample axis of
c 	      quantization, defined to be the z-axis, which is parallel
c 	      to Q: note that the x-axes of the laboratory and sample
c 	      coordinate systems are taken to be coincident. The sense
c 	      of rotation is the following: EPS is the angle FROM the
c 	      sample z-axis TO the lab z-axis rotating CCW about the
c 	      x-axis as viewed from the positive side of the x-axis.
c 	      For the lab z-axis to be aligned with the positive y-axis
c 	      of the sample, for example, EPS must be 3pi/2 radians.
c 
c 	IP, IM = complex numbers describing the polarization state
c 		   of the incident neutron relative to the lab-
c 		   oratory axis of quantization: e.g., spin "+" is
c 		   represented by IP = (1.0,0.0) and IM =
c 		   (0.0,0.0) whereas a neutron in the pure spin
c 		   "-" state is given by IP = (0.0,0.0) and IM =
c 		   (1.0,0.0). Note that in this program, the incident,
c 		   reflected, and transmitted amplitudes and intensities
c 		   are referred to the laboratory frame: a similarity
c 		   transformation is then performed on the transfer
c 		   matrix to go from the sample system, in which it was
c 		   originally formulated, to the lab system. This is
c 		   different from what is done in predecessors of this
c 		   program, such as "r6dp.f", in which the amplitudes &
c 		   intensities are rotated from lab to sample reference
c		   frame and back (with the transfer matrix correspond-
c 		   ing to the sample scattering potential remaining
c 		   unchanged in the sample coordinate system).
c
c 	ROINP = nuclear + magnetic SLD of incident medium for "+" spin
c
c 	ROINM =    "    -    "      "        "       "        "-"  "
c
c 	ROSUP =    "    -    "      "     substrate  "        "+"  "
c
c 	ROSUM =    "    -    "      "        "       "        "-"  " 
c 
c 	The parameters defined above are input into the program
c 	through the file "inpt.d".
c
c Another input file called "tro.d" contains information about each
c individual layer comprising the sample. Starting with the first
c layer encountered by the incident beam, the following quantities
c for the jth layer are supplied in the format as shown:
c
c 	T(J)  BN(J)  PN(J)  THE(J)  PHI(J) 
c
c                      .
c                      . 
c                      .
c 
c 	where
c
c 	T(J) = layer thickness in A
c
c 	BN(J) = nuclear SLD in A-2 (e.g., 8.05e-06 for Fe)
c
c 	PN(J) = magnetic SLD in A-2 (e.g., 5.085e-06 A-2 -- for Fe --
c 		corresponds to a B-field of ~ 22,000. Gauss)
c
c 	THE(J) = angle in radians that the layer magnetization
c 	         direction makes wrt the + x-axis of the sample:
c 		 note that the sample z-axis is parallel to Q
c 		 so that the sample x- and y-axes lie in the plane
c 		 of the laminar film sample. THE(J) must be defined
c 		 in the interval between zero and pi.
c
c 	PHI(J) = angle, in radians, of the projection of the layer
c 		 magnetization in the sample coordinate system's
c 		 (y,z)-plane relative to the sample y-axis. The
c 		 sense of rotation is CCW FROM the y-axis TO the
c 		 magnetization projection about the x-axis as view-
c 		 from the positive side of the x-axis. PHI(J) can
c 		 be defined in the interval between zero and 2pi.
c
c It must be noted that in the continuum reflectivity calculation
c performed by this program, Maxwell's equations apply, specifically
c the requirement that the component of the magnetic induction, B,
c normal to a boundary surface be continuous. Neither the program
c nor the wave equation itself automatically insure that this is so:
c this condition must be satisfied by appropriate selection of the
c magnetic field direction in the incident and substrate media,
c defined by the angle "EPS", and by the values of PN(J), THE(J),
c and PHI(J) specified in the input.
c
c Be aware that earlier versions of this program, such as "r6dp.f",
c do not allow for magnetic incident or substrate media AND ALSO
c require that PHI(J) be zero or pi only so that no magnetization
c in the sample is parallel to Q or normal to the plane of the film.
c
c The output files contain the spin-dependent reflectivities and
c transmissions, relative to the laboratory axis of quantization --
c which is the same in the incident and substrate media -- as follows:
c
c 	qrp2.d -- probability that the neutron will be reflected in
c 		  the plus spin state
c
c 	qrm2.d -- probability that the neutron will be reflected in
c 		  the minus spin state
c
c 	qtp2.d -- probability that the neutron will be transmitted
c 		  in the plus spin state
c
c 	qtm2.d -- probability that the neutron will be transmitted
c 		  in the minus spin state
c
c all of the above as a function of Q in A-1.
c 
c Also output are the files:
c
c 	qrpmtpms.d -- the reflectivities and transmissions, in the
c 	              above order, and their sums as a function of Q
c
c 	sum.d -- Q, sum of reflectivities and transmissions
c
c 	rpolx.d -- x-component of the polarization of the reflected
c 		   neutron vs. Q
c
c 	tpolx.d -- x-component of the polarization of the transmitted
c 		   neutron vs. Q
c
c 	rpoly.d -- y-component of the polarization of the reflected
c 		   neutron vs. Q
c
c 	tpoly.d -- y-component of the polarization of the transmitted
c 		   neutron vs. Q
c
c 	rpolz.d -- z-component of the polarization of the reflected
c 		   neutron vs. Q
c
c 	tpolz.d -- z-component of the polarization of the transmitted
c 		   neutron vs. Q
c
c 	rrem.d -- Q, Re(r"-")
c
c 	rimm.d -- Q, Im(r"-")
c
c 	rrep.d -- Q, Re(r"+")
c
c 	rimp.d -- Q, Im(r"+)
c
c 	where
c
c 		   reflectivity = Re(r)**2 + Im(r)**2
c
c *********************************************************************
c
	IMPLICIT REAL*8(A-H,O-Z)
	DIMENSION T(1000),BN(1000),PN(1000)
	DIMENSION THE(1000),PHI(1000)
	DIMENSION A(4,4),B(4,4),C(4,4)
	DIMENSION S(4),U(4),ALP(4),BET(4),GAM(4),DEL(4)
	DIMENSION CST(4,4)
	COMPLEX*16 IP,IM,CI,CR,C0,ARG1,ARG2
	COMPLEX*16 ZSP,ZSM,ZIP,ZIM,YPP,YMM,YPM,YMP
	COMPLEX*16 S,U,ALP,BET,GAM,DEL,EF,A,B,C
	COMPLEX*16 RM,RP,TP,TM,RMD,RPD,X
	COMPLEX*16 P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12
	COMPLEX*16 ARGZSP,ARGZSM,ARGZIP,ARGZIM
	COMPLEX*16 CST
	COMPLEX*16 CC,SS,SCI
	COMPLEX*16 FANGP,FANGM
	PI=4.*ATAN(1.)
c	PI=3.141592654
	CI=(0.0,1.0)
	CR=(1.0,0.0)
	C0=(0.0,0.0)
	OPEN(UNIT=10,FILE='inpt.d',STATUS='OLD',FORM='FORMATTED')
	READ(10,*)NL,NC,QS,DQ,NQ,EPS,IP,IM,ROINP,ROINM,ROSUP,ROSUM
	WRITE(*,*)NL,NC,QS,DQ,NQ,EPS,IP,IM,ROINP,ROINM,ROSUP,ROSUM
	CLOSE(10)
	OPEN(UNIT=11,FILE='tro.d',STATUS='OLD',FORM='FORMATTED')
	READ(11,*)(T(J),BN(J),PN(J),THE(J),PHI(J),J=1,NL)
	CLOSE(11)
	IF(NQ.GT.1000)GO TO 900
	IF(NL.GT.1000)GO TO 900
	OPEN(UNIT=14,FILE='qrm2.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=15,FILE='qrp2.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=16,FILE='qtm2.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=17,FILE='qtp2.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=20,FILE='qrpmtpms.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=21,FILE='sum.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=31,FILE='rrem.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=32,FILE='rimm.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=33,FILE='rrep.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=34,FILE='rimp.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=41,FILE='rpolx.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=42,FILE='tpolx.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=43,FILE='rpoly.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=44,FILE='tpoly.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=45,FILE='rpolz.d',STATUS='UNKNOWN',FORM='FORMATTED')
	OPEN(UNIT=46,FILE='tpolz.d',STATUS='UNKNOWN',FORM='FORMATTED')
	CC=CR*COS(EPS/2.)*COS(EPS/2.)
	SS=CR*SIN(EPS/2.)*SIN(EPS/2.)
	SCI=CI*COS(EPS/2.)*SIN(EPS/2.)
	DO 600 IQ=1,NQ
	DO 200 I=1,4
	DO 180 J=1,4
	B(I,J)=(0.0,0.0)
180	CONTINUE
200	CONTINUE
	B(1,1)=(1.0,0.0)
	B(2,2)=(1.0,0.0)
	B(3,3)=(1.0,0.0)
	B(4,4)=(1.0,0.0)
	Q=QS+(IQ-1)*DQ
	QP=DSQRT(Q*Q+16.*PI*ROINP)
	QM=DSQRT(Q*Q+16.*PI*ROINM)
	IF (DREAL(IP).GT.0) THEN
	  E0=QP*QP/4.
	ELSE
	  E0=QM*QM/4.
	END IF
	SUMT=0.0
	DO 400 IC=1,NC
	DO 300 L=1,NL
	SUMT=SUMT+T(L)
	ARG1=(4.*PI*(BN(L)+PN(L))-E0)
	ARG2=(4.*PI*(BN(L)-PN(L))-E0)
	S(1)=CDSQRT(ARG1)
	S(3)=CDSQRT(ARG2)
	U1NR=+1.+COS(THE(L))-SIN(THE(L))*SIN(PHI(L))
	U1NI=+SIN(THE(L))*COS(PHI(L))
	U1DR=+1.+COS(THE(L))+SIN(THE(L))*SIN(PHI(L))
	U1DI=-SIN(THE(L))*COS(PHI(L))
	U(1)=(U1NR*CR+U1NI*CI)/(U1DR*CR+U1DI*CI)
	U3NR=-2.+U1NR
	U3NI=U1NI
	U3DR=-2.+U1DR
	U3DI=U1DI
	U(3)=(U3NR*CR+U3NI*CI)/(U3DR*CR+U3DI*CI)
	S(2)=-S(1)
	S(4)=-S(3)
	U(2)=U(1)
	U(4)=U(3)
	ALP(1)=U(3)/(2.*U(3)-2.*U(1))
	BET(1)=-ALP(1)/U(3)
	GAM(1)=ALP(1)/S(1)
	DEL(1)=-ALP(1)/(U(3)*S(1))
	ALP(2)=ALP(1)
	BET(2)=-ALP(1)/U(3)
	GAM(2)=-ALP(1)/S(1)
	DEL(2)=ALP(1)/(U(3)*S(1))
	ALP(3)=-U(1)*ALP(1)/U(3)
	BET(3)=ALP(1)/U(3)
	GAM(3)=-U(1)*ALP(1)/(U(3)*S(3))
	DEL(3)=ALP(1)/(U(3)*S(3))
	ALP(4)=-U(1)*ALP(1)/U(3)
	BET(4)=ALP(1)/U(3)
	GAM(4)=U(1)*ALP(1)/(U(3)*S(3))
	DEL(4)=-ALP(1)/(U(3)*S(3))
	DO 240 I=1,4
	DO 220 J=1,4
	C(I,J)=(0.0,0.0)
	A(I,J)=(0.0,0.0)
220	CONTINUE
240	CONTINUE
	DO 260 J=1,4
	EF=CDEXP(S(J)*T(L))
	A(1,1)=A(1,1)+ALP(J)*EF
	A(1,2)=A(1,2)+BET(J)*EF
	A(1,3)=A(1,3)+GAM(J)*EF
	A(1,4)=A(1,4)+DEL(J)*EF
	A(2,1)=A(2,1)+ALP(J)*U(J)*EF
	A(2,2)=A(2,2)+BET(J)*U(J)*EF
	A(2,3)=A(2,3)+GAM(J)*U(J)*EF
	A(2,4)=A(2,4)+DEL(J)*U(J)*EF
	A(3,1)=A(3,1)+ALP(J)*S(J)*EF
	A(3,2)=A(3,2)+BET(J)*S(J)*EF
	A(3,3)=A(3,3)+GAM(J)*S(J)*EF
	A(3,4)=A(3,4)+DEL(J)*S(J)*EF
	A(4,1)=A(4,1)+ALP(J)*U(J)*S(J)*EF
	A(4,2)=A(4,2)+BET(J)*U(J)*S(J)*EF
	A(4,3)=A(4,3)+GAM(J)*U(J)*S(J)*EF
	A(4,4)=A(4,4)+DEL(J)*U(J)*S(J)*EF
260	CONTINUE
	DO 290 I=1,4
	DO 280 J=1,4
	DO 270 K=1,4
	C(I,J)=C(I,J)+A(I,K)*B(K,J)
270	CONTINUE
280	CONTINUE
290	CONTINUE
	DO 294 I=1,4
	DO 292 J=1,4
	B(I,J)=C(I,J)
292	CONTINUE
294	CONTINUE
300	CONTINUE
400	CONTINUE
	CST(1,1)=C(1,1)*CC+C(2,2)*SS+(C(2,1)-C(1,2))*SCI
	CST(1,2)=C(1,2)*CC+C(2,1)*SS+(C(2,2)-C(1,1))*SCI
	CST(2,1)=C(2,1)*CC+C(1,2)*SS+(C(1,1)-C(2,2))*SCI
	CST(2,2)=C(2,2)*CC+C(1,1)*SS+(C(1,2)-C(2,1))*SCI
	CST(1,3)=C(1,3)*CC+C(2,4)*SS+(C(2,3)-C(1,4))*SCI
	CST(1,4)=C(1,4)*CC+C(2,3)*SS+(C(2,4)-C(1,3))*SCI
	CST(2,3)=C(2,3)*CC+C(1,4)*SS+(C(1,3)-C(2,4))*SCI
	CST(2,4)=C(2,4)*CC+C(1,3)*SS+(C(1,4)-C(2,3))*SCI
	CST(3,1)=C(3,1)*CC+C(4,2)*SS+(C(4,1)-C(3,2))*SCI
	CST(3,2)=C(3,2)*CC+C(4,1)*SS+(C(4,2)-C(3,1))*SCI
	CST(4,1)=C(4,1)*CC+C(3,2)*SS+(C(3,1)-C(4,2))*SCI
	CST(4,2)=C(4,2)*CC+C(3,1)*SS+(C(3,2)-C(4,1))*SCI
	CST(3,3)=C(3,3)*CC+C(4,4)*SS+(C(4,3)-C(3,4))*SCI
	CST(3,4)=C(3,4)*CC+C(4,3)*SS+(C(4,4)-C(3,3))*SCI
	CST(4,3)=C(4,3)*CC+C(3,4)*SS+(C(3,3)-C(4,4))*SCI
	CST(4,4)=C(4,4)*CC+C(3,3)*SS+(C(3,4)-C(4,3))*SCI
	DO 480 I=1,4
	DO 470 J=1,4
	C(I,J)=CST(I,J)
470	CONTINUE
480	CONTINUE
	RMD=(0.0,0.0)
	RPD=(0.0,0.0)
	RM=(0.0,0.0)
	RP=(0.0,0.0)
	TM=(0.0,0.0)
	TP=(0.0,0.0)
	ARGZSP=(E0-4.*PI*ROSUP)
	ZSP=CI*CDSQRT(ARGZSP)
	ARGZSM=(E0-4.*PI*ROSUM)
	ZSM=CI*CDSQRT(ARGZSM)
	ARGZIP=(E0-4.*PI*ROINP)
	ZIP=CI*CDSQRT(ARGZIP)
	ARGZIM=(E0-4.*PI*ROINM)
	ZIM=CI*CDSQRT(ARGZIM)
C	PRINT*,ZSP,ZSM,ZIP,ZIM
	X=-1.*CR
	YPP=ZIP*ZSP
	YMM=ZIM*ZSM
	YPM=ZIP*ZSM
	YMP=ZIM*ZSP
	P1=ZSM*C(2,1)+X*C(4,1)+YPM*C(2,3)-ZIP*C(4,3)
	P2=ZSP*C(1,1)+X*C(3,1)-YPP*C(1,3)+ZIP*C(3,3)
	P3=ZSP*C(1,1)+X*C(3,1)+YPP*C(1,3)-ZIP*C(3,3)
	P4=ZSM*C(2,1)+X*C(4,1)-YPM*C(2,3)+ZIP*C(4,3)
	P5=ZSM*C(2,2)+X*C(4,2)+YMM*C(2,4)-ZIM*C(4,4)
	P6=ZSP*C(1,1)+X*C(3,1)-YPP*C(1,3)+ZIP*C(3,3)
	P7=ZSP*C(1,2)+X*C(3,2)+YMP*C(1,4)-ZIM*C(3,4)
	P8=ZSM*C(2,1)+X*C(4,1)-YPM*C(2,3)+ZIP*C(4,3)
	P9=ZSP*C(1,2)+X*C(3,2)-YMP*C(1,4)+ZIM*C(3,4)
	P10=ZSM*C(2,1)+X*C(4,1)-YPM*C(2,3)+ZIP*C(4,3)
	P11=ZSM*C(2,2)+X*C(4,2)-YMM*C(2,4)+ZIM*C(4,4)
	P12=ZSP*C(1,1)+X*C(3,1)-YPP*C(1,3)+ZIP*C(3,3)
	RM=RM+IP*P1*P2
	RM=RM-IP*P3*P4
	RM=RM+IM*P5*P6
	RM=RM-IM*P7*P8
	RMD=RMD+P9*P10
	RMD=RMD-P11*P12
	RM=RM/RMD
	RP=RP+RM*P9
	RP=RP+IP*P3
	RP=RP+IM*P7
	RPD=-P2
	RP=RP/RPD
	TP=C(1,1)*(IP+RP)+C(1,2)*(IM+RM)
	TP=TP+C(1,3)*ZIP*(IP-RP)+C(1,4)*ZIM*(IM-RM)
	TM=C(2,1)*(IP+RP)+C(2,2)*(IM+RM)
	TM=TM+C(2,3)*ZIP*(IP-RP)+C(2,4)*ZIM*(IM-RM)
	FANGP=ZSP*SUMT
	FANGM=ZSM*SUMT
	TP=TP*CDEXP(-FANGP)
	TM=TM*CDEXP(-FANGM)
	RM2=(DREAL(RM))**2+(DIMAG(RM))**2
	RP2=(DREAL(RP))**2+(DIMAG(RP))**2
	TP2=(DREAL(TP))**2+(DIMAG(TP))**2
	TM2=(DREAL(TM))**2+(DIMAG(TM))**2
	QV=QS+(IQ-1)*DQ
	PRXUN=2.0*DREAL(RP)*DREAL(RM)
	PRXUN=PRXUN+2.0*DIMAG(RP)*DIMAG(RM)
	PRYUN=2.0*DREAL(RP)*DIMAG(RM)
	PRYUN=PRYUN-2.0*DIMAG(RP)*DREAL(RM)
	PTXUN=2.0*DREAL(TP)*DREAL(TM)
	PTXUN=PTXUN+2.0*DIMAG(TP)*DIMAG(TM)
	PTYUN=2.0*DREAL(TP)*DIMAG(TM)
	PTYUN=PTYUN-2.0*DIMAG(TP)*DREAL(TM)
	PRX=PRXUN/(RP2+RM2)
	PRY=PRYUN/(RP2+RM2)
	PRZ=(RP2-RM2)/(RP2+RM2)
	PTX=PTXUN/(TP2+TM2)
	PTY=PTYUN/(TP2+TM2)
	PTZ=(TP2-TM2)/(TP2+TM2)
	WRITE(31,*)QV,DREAL(RM)
	WRITE(32,*)QV,DIMAG(RM)
	WRITE(33,*)QV,DREAL(RP)
	WRITE(34,*)QV,DIMAG(RP)
	SUM=RP2+RM2+TP2+TM2
	WRITE(20,*)QV,RP2,RM2,TP2,TM2,SUM
	WRITE(21,*)QV,SUM
	WRITE(14,*)QV,RM2
	WRITE(15,*)QV,RP2
	WRITE(16,*)QV,TM2
	WRITE(17,*)QV,TP2
	WRITE(41,*)QV,PRX
	WRITE(42,*)QV,PTX
	WRITE(43,*)QV,PRY
	WRITE(44,*)QV,PTY
	WRITE(45,*)QV,PRZ
	WRITE(46,*)QV,PTZ
600	CONTINUE
900	CONTINUE
	CLOSE(14)
	CLOSE(15)
	CLOSE(16)
	CLOSE(17)
	CLOSE(20)
	CLOSE(21)
	CLOSE(31)
	CLOSE(32)
	CLOSE(33)
	CLOSE(34)
	CLOSE(41)
	CLOSE(42)
	CLOSE(43)
	CLOSE(44)
	CLOSE(45)
	CLOSE(46)
	STOP
	END
