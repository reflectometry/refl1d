from math import erf, sqrt, exp

PI4 = 12.56637061435917295385
PI_180 = 0.01745329251994329576
LN256 = 5.54517744447956247533
SQRT2 = 1.41421356237309504880
SQRT2PI = 2.50662827463100050241
LOG_RESLIMIT = -6.90775527898213703123
root_12_over_2 = sqrt(3)

prange = range


def convolve_uniform(xi, yi, x, dx, y):
    left_index = 0
    N_xi = len(xi)
    N_x = len(x)
    for k in prange(N_x):
        x_k = x[k]
        # Convert 1-sigma width to 1/2 width of the region
        limit = dx[k] * root_12_over_2
        # print(f"point {x_k} +/- {limit}")
        # Find integration limits, bound by the range of the data
        left, right = max(x_k - limit, xi[0]), min(x_k + limit, xi[-1])
        if right < left:
            # Convolution does not overlap data range.
            y[k] = 0.0
            continue

        # Find the starting point for the convolution by first scanning
        # forward until we reach the next point greater than the limit
        # (we might already be there if the next output point has wider
        # resolution than the current point), then scanning backwards to
        # get to the last point before the limit. Make sure we have at
        # least one interval so that we don't have to check edge cases
        # later.
        while left_index < N_xi - 2 and xi[left_index] < left:
            left_index += 1
        while left_index > 0 and xi[left_index] > left:
            left_index -= 1

        # Set the first interval.
        total = 0.0
        right_index = left_index + 1
        x1, y1 = xi[left_index], yi[left_index]
        x2, y2 = xi[right_index], yi[right_index]

        # Subtract the excess from left interval before the left edge.
        # print(f" left {left} in {(x1, y1)}, {(x2, y2)}")
        if x1 < left:
            # Subtract the area of the rectangle from (x1, 0) to (left, y1)
            # plus 1/2 the rectangle from (x1, y1) to (left, y'),
            # where y' is y value where the line (x1, y1) to (x2, y2)
            # intersects x=left. This can be computed as follows:
            #    offset = left - x1
            #    slope = (y2 - y1)/(x2 - x1)
            #    yleft = y1 + slope*offset
            #    area = offset * y1 + offset * (yleft-y1)/2
            # It can be simplified to the following:
            #    area = offset * (y1 + slope*offset/2)
            offset = left - x1
            slope = (y2 - y1) / (x2 - x1)
            area = offset * (y1 + 0.5 * slope * offset)
            total -= area
            # print(f" left correction {area}")

        # Do trapezoidal integration up to and including the end interval
        while right_index < N_xi - 1 and x2 < right:
            # Add the current interval if it isn't empty
            if x1 != x2:
                area = 0.5 * (y1 + y2) * (x2 - x1)
                total += area
                # print(f" adding {(x1,y1)}, {(x2, y2)} as {area}")
            # Move to the next interval
            right_index += 1
            x1, y1, x2, y2 = x2, y2, xi[right_index], yi[right_index]
        if x1 != x2:
            area = 0.5 * (y1 + y2) * (x2 - x1)
            total += area
            # print(f" adding final {(x1,y1)}, {(x2, y2)} as {area}")

        # Subtract the excess from the right interval after the right edge.
        # print(f" right {right} in {(x1, y1)}, {(x2, y2)}")
        if x2 > right:
            # Expression for area to subtract using rectangles is as follows:
            #    offset = x2 - right
            #    slope = (y2 - y1)/(x2 - x1)
            #    yright = y2 - slope*offset
            #    area = -(offset * yright + offset * (y2-yright)/2)
            # It can be simplified to the following:
            #    area = -offset * (y2 - slope*offset/2)
            offset = x2 - right
            slope = (y2 - y1) / (x2 - x1)
            area = offset * (y2 - 0.5 * slope * offset)
            total -= area
            # print(f" right correction {area}")

        # Normalize by interval length
        if left < right:
            # print(f" normalize by length {right} - {left}")
            y[k] = total / (right - left)
        elif x1 < x2:
            # If dx = 0 using the value interpolated at x (with left=right=x).
            # print(f" dirac delta at {left} = {right} in {(x1, y1)}, {(x2, y2)}")
            offset = left - x1
            slope = (y2 - y1) / (x2 - x1)
            y[k] = y1 + slope * offset
        else:
            # At an empty interval in the theory function. Average the y.
            # print(f" empty interval with {left} = {right} in {(x1, y1)}, {(x2, y2)}")
            y[k] = 0.5 * (y1 + y2)


def convolve_gaussian_point(xin, yin, k, n, xo, limit, sigma):
    two_sigma_sq = 2.0 * sigma * sigma
    # double z, Glo, erflo, erfmin, y

    z = xo - xin[k]
    Glo = exp(-z * z / two_sigma_sq)
    erfmin = erflo = erf(-z / (SQRT2 * sigma))
    y = 0.0
    # /* printf("%5.3f: (%5.3f,%11.5g)",xo,xin[k],yin[k]); */
    while k < n - 1:
        k += 1
        if xin[k] != xin[k - 1]:
            # /* No additional contribution from duplicate points. */

            # /* Compute the next endpoint */
            zhi = xo - xin[k]
            Ghi = exp(-zhi * zhi / two_sigma_sq)
            erfhi = erf(-zhi / (SQRT2 * sigma))
            m = (yin[k] - yin[k - 1]) / (xin[k] - xin[k - 1])
            b = yin[k] - m * xin[k]

            # /* Add the integrals. */
            y += 0.5 * (m * xo + b) * (erfhi - erflo) - sigma / SQRT2PI * m * (Ghi - Glo)

            # /* Debug computation failures. */
            # if isnan(y) {
            #     print("NaN from %d: zhi=%g, Ghi=%g, erfhi=%g, m=%g, b=%g\n",
            #          % (k,zhi,Ghi,erfhi,m,b))
            # }

            # /* Save the endpoint for next trapezoid. */
            Glo = Ghi
            erflo = erfhi

            # /* Check if we've calculated far enough */
            if xin[k] >= xo + limit:
                break

    # /* printf(" (%5.3f,%11.5g)",xin[k<n?k:n-1],yin[k<n?k:n-1]); */

    # /* Normalize by the area of the truncated gaussian */
    # /* At this point erflo = erfmax */
    # /* printf ("---> %11.5g\n",2*y/(erflo-erfmin)); */
    return 2 * y / (erflo - erfmin)


def convolve_gaussian(xin, yin, x, dx, y):
    # size_t in,out;
    Nin = len(xin)
    Nout = len(x)

    # /* FIXME fails if xin are not sorted; slow if x not sorted */
    # assert(Nin>1)

    # /* Scan through all x values to be calculated */
    # /* Re: omp, each thread is going through the entire input array,
    # * independently, computing the resolution from the neighbourhood
    # * around its individual output points.  The firstprivate(in)
    # * clause sets each thread to keep its own copy of in, initialized
    # * at in's initial value of zero.  The "schedule(static,1)" clause
    # * puts neighbouring points in separate threads, which is a benefit
    # * since there will be less backtracking if resolution width increases
    # * from point to point.  Because the schedule is static, this does not
    # * significantly increase the parallelization overhead.  Because the
    # * threads are operating on interleaved points, there should be fewer cache
    # * misses than if each thread were given different stretches of x to
    # * convolve.
    # */
    k_in = 0

    for k_out in range(Nout):
        # /* width of resolution window for x is w = 2 dx^2. */
        sigma = dx[k_out]
        xo = x[k_out]
        limit = sqrt(-2.0 * sigma * sigma * LOG_RESLIMIT)

        # // if (out%20==0)

        # /* Line up the left edge of the convolution window */
        # /* It is probably forward from the current position, */
        # /* but if the next dx is a lot higher than the current */
        # /* dx or if the x are not sorted, then it may be before */
        # /* the current position. */
        # /* FIXME verify that the convolution window is just right */
        while k_in < Nin - 1 and xin[k_in] < xo - limit:
            k_in += 1
        while k_in > 0 and xin[k_in] > xo - limit:
            k_in -= 1

        # /* Special handling to avoid 0/0 for w=0. */
        if sigma > 0.0:
            y[k_out] = convolve_gaussian_point(xin, yin, k_in, Nin, xo, limit, sigma)
        elif k_in < Nin - 1:
            # /* Linear interpolation */
            m = (yin[k_in + 1] - yin[k_in]) / (xin[k_in + 1] - xin[k_in])
            b = yin[k_in] - m * xin[k_in]
            y[k_out] = m * xo + b
        elif k_in > 0:
            # /* Linear extrapolation */
            m = (yin[k_in] - yin[k_in - 1]) / (xin[k_in] - xin[k_in - 1])
            b = yin[k_in] - m * xin[k_in]
            y[k_out] = m * xo + b
        else:
            # /* Can't happen because there is more than one point in xin. */
            # assert(Nin>1)
            pass
