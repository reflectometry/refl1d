import math

Z_EPS = 1e-6


def align_magnetic(d, sigma, rho, irho, dM, sigmaM, rhoM, thetaM, output_flat):
    # ignoring thickness d on the first and last layers
    # ignoring interface width sigma on the last layer
    nlayers = len(d)
    nlayersM = len(dM)
    noutput = nlayers + nlayersM

    # making sure there are at least two layers
    if nlayers < 2 or nlayersM < 2:
        raise ValueError("only works with more than one layer")

    output = output_flat
    magnetic = 0  # current magnetic layer index
    nuclear = 0  # current nuclear layer index
    z = 0.0  # current interface depth
    next_z = 0.0  # next nuclear interface
    next_zM = 0.0  # next magnetic interface
    # int active = 3# active interfaces, active&0x1 for nuclear, active&0x2 for magnetic

    k = 0  # current output layer index
    while True:
        # repeat over all nuclear/magnetic layers
        if k == noutput:
            return -1  # exceeds capacity of output

        # printf("%d: %d %d %g %g %g\n", k, nuclear, magnetic, z, next_z, next_zM)
        # printf("%g %g %g %g\n", rho[nuclear], irho[nuclear], rhoM[magnetic], thetaM[magnetic])
        # printf("%g %g %g %g\n", d[nuclear], sigma[nuclear], dM[magnetic], sigmaM[magnetic])

        # Set the scattering strength using the current parameters
        output[k][2] = rho[nuclear]
        output[k][3] = irho[nuclear]
        output[k][4] = rhoM[magnetic]
        output[k][5] = thetaM[magnetic]

        # Check if we are at the last layer for both nuclear and magnetic
        # If so set thickness and interface width to zero.  We are doing a
        # center of the loop exit in order to make sure that the final layer
        # is added.

        if magnetic == nlayersM - 1 and nuclear == nlayers - 1:
            output[k][0] = 0.0
            output[k][1] = 0.0
            k += 1
            break

        # Determine if we are adding the nuclear or the magnetic interface next,
        # or possibly both.  The order of the conditions is important.
        #
        # Note: the final value for next_z/next_zM is not defined.  Rather than
        # checking if we are on the last layer we simply add the value of the
        # last thickness to z, which may be 0, nan, inf, or anything else.  This
        # doesn't affect the algorithm since we don't look at next_z when we are
        # on the final nuclear layer or next_zM when we are on the final magnetic
        # layer.
        #
        # Note: averaging nearly aligned interfaces can lead to negative thickness
        # Consider nuc = [1-a, 0, 1] and mag = [1+a, 1, 1] for 2a < Z_EPS.
        # On the first step we set next_z to 1-a, next_zM to 1+a and z to the
        # average of 1-a and 1+a, which is 1.  On the second step next_z is
        # still 1-a, so the thickness next_z - z = -a. Since a is tiny we can just
        # pretend that -a == zero by setting thickness to fmax(next_z - z, 0.0).

        if nuclear == nlayers - 1:
            # No more nuclear layers... play out the remaining magnetic layers.
            output[k][0] = max(next_zM - z, 0.0)
            output[k][1] = sigmaM[magnetic]
            magnetic += 1
            next_zM += dM[magnetic]
        elif magnetic == nlayersM - 1:
            # No more magnetic layers... play out the remaining nuclear layers.
            output[k][0] = max(next_z - z, 0.0)
            output[k][1] = sigma[nuclear]
            nuclear += 1
            next_z += d[nuclear]
        elif math.fabs(next_z - next_zM) < Z_EPS and math.fabs(sigma[nuclear] - sigmaM[magnetic]) < Z_EPS:
            # Matching nuclear/magnetic boundary, with almost identical interfaces.
            # Increment both nuclear and magnetic layers.
            output[k][0] = max(0.5 * (next_z + next_zM) - z, 0.0)
            output[k][1] = 0.5 * (sigma[nuclear] + sigmaM[magnetic])
            nuclear += 1
            next_z += d[nuclear]
            magnetic += 1
            next_zM += dM[magnetic]
        elif next_zM < next_z:
            # Magnetic boundary comes before nuclear boundary, so increment magnetic.
            output[k][0] = max(next_zM - z, 0.0)
            output[k][1] = sigmaM[magnetic]
            magnetic += 1
            next_zM += dM[magnetic]
        else:
            # Nuclear boundary comes before magnetic boundary
            # OR nuclear and magnetic boundaries match but interfaces are different.
            # so increment nuclear.
            output[k][0] = max(next_z - z, 0.0)
            output[k][1] = sigma[nuclear]
            nuclear += 1
            next_z += d[nuclear]

        z += output[k][0]
        k += 1

    return k


def contract_mag(d, sigma, rho, irho, rhoM, thetaM, dA):
    n = len(d)
    m = n - 1  # /* last middle layer */
    i = newi = 1  # /* Skip the substrate */
    while i < m:
        # /* Get ready for the next layer */
        # /* Accumulation of the first row happens in the inner loop */
        dz = 0
        rhoarea = irhoarea = rhoMpara_area = rhoMperp_area = thetaM_area = 0.0
        rholo = rhohi = rho[i]
        irholo = irhohi = irho[i]

        # /*
        #  * averaged thetaM is from atan2 (when rhoM is nonzero)
        #  * which returns values in the range -180 to 180,
        #  * so we keep track of the phase offset of the input
        #  * to match it afterward
        #  */
        thetaM_phase_offset = (thetaM[i] + 180.0) // 360.0

        # /* Pre-calculate projections */
        rhoM_sign = math.copysign(1, rhoM[i])
        thetaM_radians = thetaM[i] * math.pi / 180.0
        rhoMpara = rhoM[i] * math.cos(thetaM_radians)
        rhoMperp = rhoM[i] * math.sin(thetaM_radians)
        # /*
        #  * Note that mpara indicates M when theta_M = 0,
        #  * and mperp indicates M when theta_M = 90,
        #  * but in most cases M is parallel to H when theta_M = 270
        #  * and Aguide = 270
        #  */
        mparalo = mparahi = rhoMpara
        mperplo = mperphi = rhoMperp

        # /* Accumulate slices into layer */
        while i < m:
            # /* Accumulate next slice */
            dz += d[i]
            rhoarea += d[i] * rho[i]
            irhoarea += d[i] * irho[i]
            # /* thetaM_area is only used if rhoM is zero */
            thetaM_area += d[i] * thetaM[i]
            # /* Use pre-calculated next values */
            rhoMpara_area += rhoMpara * d[i]
            rhoMperp_area += rhoMperp * d[i]

            # /* If no more slices or sigma != 0, break immediately */
            i += 1
            if i == n or sigma[i - 1] != 0.0:
                break

            # /* If next slice exceeds limit then break */
            if rho[i] < rholo:
                rholo = rho[i]
            if rho[i] > rhohi:
                rhohi = rho[i]
            if (rhohi - rholo) * (dz + d[i]) > dA:
                break

            if irho[i] < irholo:
                irholo = irho[i]
            if irho[i] > irhohi:
                irhohi = irho[i]
            if (irhohi - irholo) * (dz + d[i]) > dA:
                break

            # /* Pre-calculate projections of next layer */
            thetaM_radians = thetaM[i] * math.pi / 180.0
            rhoMpara = rhoM[i] * math.cos(thetaM_radians)
            rhoMperp = rhoM[i] * math.sin(thetaM_radians)

            # /* If next slice is wrapped in phase, break */
            if (thetaM[i] + 180.0) // 360.0 != thetaM_phase_offset:
                break

            # /* If next slice has different sign for rhoM, break */
            if math.copysign(1, rhoM[i]) != rhoM_sign:
                break

            if rhoMpara < mparalo:
                mparalo = rhoMpara
            if rhoMpara > mparahi:
                mparahi = rhoMpara
            if (mparahi - mparalo) * (dz + d[i]) > dA:
                break

            if rhoMperp < mperplo:
                mperplo = rhoMperp
            if rhoMperp > mperphi:
                mperphi = rhoMperp
            if (mperphi - mperplo) * (dz + d[i]) > dA:
                break

        assert newi < m
        d[newi] = dz
        if dz == 0:
            rho[newi] = rho[i - 1]
            irho[newi] = irho[i - 1]
            rhoM[newi] = rhoM[i - 1]
            thetaM[newi] = thetaM[i - 1]
        else:
            rho[newi] = rhoarea / dz
            irho[newi] = irhoarea / dz
            mean_rhoMpara = rhoMpara_area / dz
            mean_rhoMperp = rhoMperp_area / dz
            mean_rhoM = math.sqrt(mean_rhoMpara**2 + mean_rhoMperp**2) * rhoM_sign
            if mean_rhoM == 0:
                # /* If rhoM is zero, then thetaM is meaningless: use plain average */
                thetaM[newi] = thetaM_area / dz
            else:
                # /* Otherwise, calculate the mean thetaM
                #  * invert the sign of components if rhoM is negative, before atan2 */
                thetaM_from_mean = math.atan2(mean_rhoMperp * rhoM_sign, mean_rhoMpara * rhoM_sign) * 180.0 / math.pi
                thetaM_from_mean += 360.0 * thetaM_phase_offset
                thetaM[newi] = thetaM_from_mean
            rhoM[newi] = mean_rhoM
        sigma[newi] = sigma[i - 1]

        newi += 1

    # /* Save the last layer */
    # /* Last layer uses surface values */
    rho[newi] = rho[n - 1]
    irho[newi] = irho[n - 1]
    rhoM[newi] = rhoM[n - 1]
    thetaM[newi] = thetaM[n - 1]
    # /* No interface for final layer */
    newi += 1

    return newi


def contract_by_area(d, sigma, rho, irho, dA):
    n = len(d)
    i = newi = 1  # /* Skip the substrate */
    while i < n:
        # /* Get ready for the next layer */
        # /* Accumulation of the first row happens in the inner loop */
        dz = rhoarea = irhoarea = 0.0
        rholo = rhohi = rho[i]
        irholo = irhohi = irho[i]

        # /* Accumulate slices into layer */
        while True:
            # /* Accumulate next slice */
            dz += d[i]
            rhoarea += d[i] * rho[i]
            irhoarea += d[i] * irho[i]

            # /* If no more slices or sigma != 0, break immediately */
            i += 1
            if i == n or sigma[i - 1] != 0.0:
                break

            # /* If next slice won't fit, break */
            if rho[i] < rholo:
                rholo = rho[i]
            if rho[i] > rhohi:
                rhohi = rho[i]
            if (rhohi - rholo) * (dz + d[i]) > dA:
                break

            if irho[i] < irholo:
                irholo = irho[i]
            if irho[i] > irhohi:
                irhohi = irho[i]
            if (irhohi - irholo) * (dz + d[i]) > dA:
                break

        # /* dz is only going to be zero if there is a forced break due to
        # * sigma, or if we are accumulating a substrate.  In either case,
        # * we want to accumulate the zero length layer
        # */
        # /* if (dz == 0) continue; */

        # /* Save the layer */
        assert newi < n
        d[newi] = dz
        if i == n:
            # /* printf("contract: adding final sld at %d\n",newi); */
            # /* Last layer uses surface values */
            rho[newi] = rho[n - 1]
            irho[newi] = irho[n - 1]
            # /* No interface for final layer */
        else:
            # /* Middle layers uses average values */
            rho[newi] = rhoarea / dz
            irho[newi] = irhoarea / dz
            sigma[newi] = sigma[i - 1]
        # /* First layer uses substrate values */
        newi += 1

    return newi
