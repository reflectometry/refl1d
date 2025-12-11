import sys
from numpy import pi, sin, cos, radians, sqrt, exp, fabs

EPS = sys.float_info.epsilon
M_PI = pi
PI4 = 4.0e-6 * pi
B2SLD = 2.31604654  # Scattering factor for B field 1e-6

prange = range


def calculate_U1_U3_single(H, rhoM, thetaM, Aguide, U1, U3, index):
    # thetaM should be in radians,
    # Aguide in degrees.
    # phiH = (Aguide - 270.0)*M_PI/180.0;
    AG = Aguide * M_PI / 180.0  # Aguide in radians
    # thetaH = M_PI_2; # by convention, H is in y-z plane so theta = pi/2

    sld_h = B2SLD * H
    sld_m_x = rhoM[index] * cos(thetaM[index])
    sld_m_y = rhoM[index] * sin(thetaM[index])
    sld_m_z = 0.0  # by Maxwell's equations, H_demag = mz so we'll just cancel it here
    # The purpose of AGUIDE is to rotate the z-axis of the sample coordinate
    # system so that it is aligned with the quantization axis z, defined to be
    # the direction of the magnetic field outside the sample.

    new_my = sld_m_z * sin(AG) + sld_m_y * cos(AG)
    new_mz = sld_m_z * cos(AG) - sld_m_y * sin(AG)
    sld_m_y = new_my
    sld_m_z = new_mz
    sld_h_x = 0.0
    sld_h_y = 0.0
    sld_h_z = sld_h
    # Then, don't rotate the transfer matrix!!
    # Aguide = 0.0;

    sld_b_x = sld_h_x + sld_m_x
    sld_b_y = sld_h_y + sld_m_y
    sld_b_z = sld_h_z + sld_m_z

    # avoid divide-by-zero:
    sld_b_x += EPS * (sld_b_x == 0)
    sld_b_y += EPS * (sld_b_y == 0)

    # add epsilon to y, to avoid divide by zero errors?
    sld_b = sqrt(pow(sld_b_x, 2) + pow(sld_b_y, 2) + pow(sld_b_z, 2))
    u1_num = complex(sld_b + sld_b_x - sld_b_z, sld_b_y)
    u1_den = complex(sld_b + sld_b_x + sld_b_z, -sld_b_y)
    u3_num = complex(-sld_b + sld_b_x - sld_b_z, sld_b_y)
    u3_den = complex(-sld_b + sld_b_x + sld_b_z, -sld_b_y)

    U1[index] = u1_num / u1_den
    U3[index] = u3_num / u3_den
    rhoM[index] = sld_b


def calculate_u1_u3(H, rhoM, thetaM, Aguide, u1, u3):
    """
    array version - rhoM, thetaM, u1 and u3 are arrays
    rhoM, u1 and u3 are modified in-place
    """
    for i in range(len(rhoM)):
        calculate_U1_U3_single(H, rhoM, thetaM, Aguide, u1, u3, i)


B2SLD = 2.31604654  # Scattering factor for B field 1e-6


def Cr4xa(N, D, SIGMA, IP, RHO, IRHO, RHOM, U1, U3, KZ, POINT, YA, YB, YC, YD):
    EPS = 1e-10

    if KZ <= -1.0e-10:
        L = N - 1
        STEP = -1
        SIGMA_OFFSET = -1
    elif KZ >= 1.0e-10:
        L = 0
        STEP = 1
        SIGMA_OFFSET = 0
    else:
        YA[POINT] = -1.0
        YB[POINT] = 0.0
        YC[POINT] = 0.0
        YD[POINT] = -1.0
        return

    #    Changing the target KZ is equivalent to subtracting the fronting
    #    medium SLD.

    # IP = 1 specifies polarization of the incident beam I+
    # IP = -1 specifies polarization of the incident beam I-
    E0 = KZ * KZ + PI4 * (RHO[L] + IP * RHOM[L])

    Z = 0.0
    if N > 1:
        # chi in layer 1
        LP = L + STEP
        # Branch selection:  the -sqrt below for S1 and S3 will be
        #     +Imag for KZ > Kcrit,
        #     -Real for KZ < Kcrit
        # which covers the S1, S3 waves allowed by the boundary conditions in the
        # fronting and backing medium:
        # either traveling forward (+Imag) or decaying exponentially forward (-Real).
        # The decaying exponential only occurs for the transmitted forward wave in the backing:
        # the root +iKz is automatically chosen for the incident wave in the fronting.
        #
        # In the fronting, the -S1 and -S3 waves are either traveling waves backward (-Imag)
        # or decaying along the -z reflection direction (-Real) * (-z) = (+Real*z).
        # NB: This decaying reflection only occurs when the reflected wave is below Kcrit
        # while the incident wave is above Kcrit, so it only happens for spin-flip from
        # minus to plus (lower to higher potential energy) and the observed R-+ will
        # actually be zero at large distances from the interface.
        #
        # In the backing, the -S1 and -S3 waves are explicitly set to be zero amplitude
        # by the boundary conditions (neutrons only incident in the fronting medium - no
        # source of neutrons below).
        #
        S1L = -sqrt(complex(PI4 * (RHO[L] + RHOM[L]) - E0, -PI4 * (fabs(IRHO[L]) + EPS)))
        S3L = -sqrt(complex(PI4 * (RHO[L] - RHOM[L]) - E0, -PI4 * (fabs(IRHO[L]) + EPS)))
        S1LP = -sqrt(complex(PI4 * (RHO[LP] + RHOM[LP]) - E0, -PI4 * (fabs(IRHO[LP]) + EPS)))
        S3LP = -sqrt(complex(PI4 * (RHO[LP] - RHOM[LP]) - E0, -PI4 * (fabs(IRHO[LP]) + EPS)))
        SIGMAL = SIGMA[L + SIGMA_OFFSET]

        if abs(U1[L]) <= 1.0:
            # then Bz >= 0
            # BL and GL are zero in the fronting.
            pass
        else:
            # then Bz < 0: flip!
            # This is probably impossible, since Bz defines the +z direction
            # in the fronting medium, but just in case...
            SSWAP = S1L
            S1L = S3L
            S3L = SSWAP  # swap S3 and S1

        if abs(U1[LP]) <= 1.0:
            # then Bz >= 0
            BLP = U1[LP]
            GLP = 1.0 / U3[LP]
        else:
            # then Bz < 0: flip!
            BLP = U3[LP]
            GLP = 1.0 / U1[LP]
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP  # swap S3 and S1

        DELTA = 0.5 / (1.0 - (BLP * GLP))

        FS1S1 = S1L / S1LP
        FS1S3 = S1L / S3LP
        FS3S1 = S3L / S1LP
        FS3S3 = S3L / S3LP

        B11 = DELTA * 1.0 * (1.0 + FS1S1)
        B12 = DELTA * 1.0 * (1.0 - FS1S1) * exp(2.0 * S1L * S1LP * SIGMAL * SIGMAL)
        B13 = DELTA * -GLP * (1.0 + FS3S1)
        B14 = DELTA * -GLP * (1.0 - FS3S1) * exp(2.0 * S3L * S1LP * SIGMAL * SIGMAL)

        B21 = DELTA * 1.0 * (1.0 - FS1S1) * exp(2.0 * S1L * S1LP * SIGMAL * SIGMAL)
        B22 = DELTA * 1.0 * (1.0 + FS1S1)
        B23 = DELTA * -GLP * (1.0 - FS3S1) * exp(2.0 * S3L * S1LP * SIGMAL * SIGMAL)
        B24 = DELTA * -GLP * (1.0 + FS3S1)

        B31 = DELTA * -BLP * (1.0 + FS1S3)
        B32 = DELTA * -BLP * (1.0 - FS1S3) * exp(2.0 * S1L * S3LP * SIGMAL * SIGMAL)
        B33 = DELTA * 1.0 * (1.0 + FS3S3)
        B34 = DELTA * 1.0 * (1.0 - FS3S3) * exp(2.0 * S3L * S3LP * SIGMAL * SIGMAL)

        B41 = DELTA * -BLP * (1.0 - FS1S3) * exp(2.0 * S1L * S3LP * SIGMAL * SIGMAL)
        B42 = DELTA * -BLP * (1.0 + FS1S3)
        B43 = DELTA * 1.0 * (1.0 - FS3S3) * exp(2.0 * S3L * S3LP * SIGMAL * SIGMAL)
        B44 = DELTA * 1.0 * (1.0 + FS3S3)

        Z += D[LP]
        L = LP

    #    Process the loop once for each interior layer, either from
    #    front to back or back to front.
    for I in range(1, N - 1):
        LP = L + STEP
        S1L = S1LP  # copy from the layer before
        S3L = S3LP
        GL = GLP
        BL = BLP
        S1LP = -sqrt(complex(PI4 * (RHO[LP] + RHOM[LP]) - E0, -PI4 * (fabs(IRHO[LP]) + EPS)))
        S3LP = -sqrt(complex(PI4 * (RHO[LP] - RHOM[LP]) - E0, -PI4 * (fabs(IRHO[LP]) + EPS)))
        SIGMAL = SIGMA[L + SIGMA_OFFSET]

        if abs(U1[LP]) <= 1.0:
            # then Bz >= 0
            BLP = U1[LP]
            GLP = 1.0 / U3[LP]
        else:
            # then Bz < 0: flip!
            BLP = U3[LP]
            GLP = 1.0 / U1[LP]
            SSWAP = S1LP
            S1LP = S3LP
            S3LP = SSWAP  # swap S3 and S1

        DELTA = 0.5 / (1.0 - (BLP * GLP))
        DBB = (BL - BLP) * DELTA  # multiply by delta here?
        DBG = (1.0 - BL * GLP) * DELTA
        DGB = (1.0 - GL * BLP) * DELTA
        DGG = (GL - GLP) * DELTA

        ES1L = exp(S1L * Z)
        ENS1L = 1.0 / ES1L
        ES1LP = exp(S1LP * Z)
        ENS1LP = 1.0 / ES1LP
        ES3L = exp(S3L * Z)
        ENS3L = 1.0 / ES3L
        ES3LP = exp(S3LP * Z)
        ENS3LP = 1.0 / ES3LP

        FS1S1 = S1L / S1LP
        FS1S3 = S1L / S3LP
        FS3S1 = S3L / S1LP
        FS3S3 = S3L / S3LP

        A11 = A22 = DBG * (1.0 + FS1S1)
        A11 *= ES1L * ENS1LP
        A22 *= ENS1L * ES1LP
        A12 = A21 = DBG * (1.0 - FS1S1) * exp(2.0 * S1L * S1LP * SIGMAL * SIGMAL)
        A12 *= ENS1L * ENS1LP
        A21 *= ES1L * ES1LP
        A13 = A24 = DGG * (1.0 + FS3S1)
        A13 *= ES3L * ENS1LP
        A24 *= ENS3L * ES1LP
        A14 = A23 = DGG * (1.0 - FS3S1) * exp(2.0 * S3L * S1LP * SIGMAL * SIGMAL)
        A14 *= ENS3L * ENS1LP
        A23 *= ES3L * ES1LP

        A31 = A42 = DBB * (1.0 + FS1S3)
        A31 *= ES1L * ENS3LP
        A42 *= ENS1L * ES3LP
        A32 = A41 = DBB * (1.0 - FS1S3) * exp(2.0 * S1L * S3LP * SIGMAL * SIGMAL)
        A32 *= ENS1L * ENS3LP
        A41 *= ES1L * ES3LP
        A33 = A44 = DGB * (1.0 + FS3S3)
        A33 *= ES3L * ENS3LP
        A44 *= ENS3L * ES3LP
        A34 = A43 = DGB * (1.0 - FS3S3) * exp(2.0 * S3L * S3LP * SIGMAL * SIGMAL)
        A34 *= ENS3L * ENS3LP
        A43 *= ES3L * ES3LP

        #    Matrix update B=A*B
        C1 = A11 * B11 + A12 * B21 + A13 * B31 + A14 * B41
        C2 = A21 * B11 + A22 * B21 + A23 * B31 + A24 * B41
        C3 = A31 * B11 + A32 * B21 + A33 * B31 + A34 * B41
        C4 = A41 * B11 + A42 * B21 + A43 * B31 + A44 * B41
        B11 = C1
        B21 = C2
        B31 = C3
        B41 = C4

        C1 = A11 * B12 + A12 * B22 + A13 * B32 + A14 * B42
        C2 = A21 * B12 + A22 * B22 + A23 * B32 + A24 * B42
        C3 = A31 * B12 + A32 * B22 + A33 * B32 + A34 * B42
        C4 = A41 * B12 + A42 * B22 + A43 * B32 + A44 * B42
        B12 = C1
        B22 = C2
        B32 = C3
        B42 = C4

        C1 = A11 * B13 + A12 * B23 + A13 * B33 + A14 * B43
        C2 = A21 * B13 + A22 * B23 + A23 * B33 + A24 * B43
        C3 = A31 * B13 + A32 * B23 + A33 * B33 + A34 * B43
        C4 = A41 * B13 + A42 * B23 + A43 * B33 + A44 * B43
        B13 = C1
        B23 = C2
        B33 = C3
        B43 = C4

        C1 = A11 * B14 + A12 * B24 + A13 * B34 + A14 * B44
        C2 = A21 * B14 + A22 * B24 + A23 * B34 + A24 * B44
        C3 = A31 * B14 + A32 * B24 + A33 * B34 + A34 * B44
        C4 = A41 * B14 + A42 * B24 + A43 * B34 + A44 * B44
        B14 = C1
        B24 = C2
        B34 = C3
        B44 = C4

        Z += D[LP]
        L = LP

    #    Done computing B = A(N)*...*A(2)*A(1)*I
    DETW = B44 * B22 - B24 * B42

    #    Calculate reflectivity coefficients specified by POLSTAT
    # IP = +1 fills in ++, +-, -+, --; IP = -1 only fills in -+, --.
    if IP > 0:
        YA[POINT] = (B24 * B41 - B21 * B44) / DETW  # ++
        YB[POINT] = (B21 * B42 - B41 * B22) / DETW  # +-
    YC[POINT] = (B24 * B43 - B23 * B44) / DETW  # -+
    YD[POINT] = (B23 * B42 - B43 * B22) / DETW  # --


def magnetic_amplitude(d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index, Ra, Rb, Rc, Rd):
    """
    python version of calculation
    implicit returns: Ra, Rb, Rc, Rd
    """
    # assert rho_index is None
    layers = len(d)
    points = len(KZ)

    # plus polarization must be before minus polarization because it
    # fills in all R++, R+-, R-+, R--, but minus polarization only fills
    # in R-+, R--.
    for i in prange(points):
        Cr4xa(layers, d, sigma, 1.0, rho, irho, rhoM, u1, u3, KZ[i], i, Ra, Rb, Rc, Rd)

    # minus polarization
    for i in prange(points):
        Cr4xa(layers, d, sigma, -1.0, rho, irho, rhoM, u1, u3, KZ[i], i, Ra, Rb, Rc, Rd)


BASE_GUIDE_ANGLE = 270.0


def calculate_u1_u3_py(H, rhoM, thetaM, Aguide):
    rotate_M = True

    thetaM = radians(thetaM)
    phiH = radians(Aguide - BASE_GUIDE_ANGLE)
    thetaH = pi / 2.0  # by convention, H is in y-z plane so theta = pi/2

    sld_h = B2SLD * H
    sld_m_x = rhoM * cos(thetaM)
    sld_m_y = rhoM * sin(thetaM)
    sld_m_z = 0.0  # by Maxwell's equations, H_demag = mz so we'll just cancel it here
    # The purpose of AGUIDE is to rotate the z-axis of the sample coordinate
    # system so that it is aligned with the quantization axis z, defined to be
    # the direction of the magnetic field outside the sample.
    if rotate_M:
        # rotate the M vector instead of the transfer matrix!
        # First, rotate the M vector about the x axis:
        new_my = sld_m_z * sin(radians(Aguide)) + sld_m_y * cos(radians(Aguide))
        new_mz = sld_m_z * cos(radians(Aguide)) - sld_m_y * sin(radians(Aguide))
        sld_m_y, sld_m_z = new_my, new_mz
        sld_h_x = sld_h_y = 0.0
        sld_h_z = sld_h
        # Then, don't rotate the transfer matrix
        Aguide = 0.0
    else:
        sld_h_x = sld_h * cos(thetaH)  # zero
        sld_h_y = sld_h * sin(thetaH) * cos(phiH)
        sld_h_z = sld_h * sin(thetaH) * sin(phiH)

    sld_b_x = sld_h_x + sld_m_x
    sld_b_y = sld_h_y + sld_m_y
    sld_b_z = sld_h_z + sld_m_z

    # avoid divide-by-zero:
    sld_b_x += EPS * (sld_b_x == 0)
    sld_b_y += EPS * (sld_b_y == 0)

    # add epsilon to y, to avoid divide by zero errors?
    sld_b = sqrt(sld_b_x**2 + sld_b_y**2 + sld_b_z**2)
    u1_num = +sld_b + sld_b_x + 1j * sld_b_y - sld_b_z
    u1_den = +sld_b + sld_b_x - 1j * sld_b_y + sld_b_z
    u3_num = -sld_b + sld_b_x + 1j * sld_b_y - sld_b_z
    u3_den = -sld_b + sld_b_x - 1j * sld_b_y + sld_b_z

    u1 = u1_num / u1_den
    u3 = u3_num / u3_den
    # print "u1", u1
    # print "u3", u3
    return sld_b, u1, u3
