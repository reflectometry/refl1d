def convolve_point_sampled(Nin, xin, yin, Np, xp, yp, xo, dx, _in):
    # Walk the theory spline and the resolution spline together, computing
    # the integral of the pairs of line segments.  Since the spline knots
    # do not necessarily correspond, we need to treat each individual segment
    # in both curves piece-wise, integrating from knot to knot in the union
    # of the two knot sets.

    # Need an undefined value in case the algorithm is broken and we don't
    # initialize the initial lines.  This would ideal be NaN, but that isn't
    # available in MSVC, so set it to something very large.  In order to avoid
    # normalizing that very large value to 1., only set the theory line to
    # undefined, and set the resolution line to 0.
    # const double undefined = 0./0.;
    undefined = 1e308
    m1 = undefined
    b1 = undefined
    m2 = 0.0
    b2 = 0.0
    sum = 0.0
    norm = 0.0
    # size_t p;
    # double delta,delta2,delta3;
    # double x, next_x, next_xin, next_xp;

    # Set the target start of the integral to xin[_in].  This may or may
    # not be beyond the end of the resolution function, depending on whether
    # we are at the left edge of the data, or somewhere inside.
    next_xin = xin[_in]

    # Set p to the point just before next_xin
    for p in range(1, Np):
        if xo + dx * xp[p] > next_xin:
            break

    p -= 1
    next_xp = xo + dx * xp[p]

    # Choose the larger of next_xp and next_xin as the starting point of the
    # integral.  This will force us to step both xin and xp in the first
    # iteration of the loop, computing new slope/intercepts for both lines.
    # If the theory line extends beyond the resolution function, then we will
    # be called with xin[_in] before the first resolution point, and next_xp
    # will be bigger than next_xin.  If the resolution extends beyond the
    # end of the theory, then the next_xp will be set before next_xin, and
    # next_xin will be the start of the integral.  The integral ends when
    # either the resolution or the theory runs out.
    #
    # We are tracking the area under the resolution as well as the area
    # under the product of theory and resolution so that we can properly
    # normalize the data when less than the full resolution is included.
    # This means that we do not need to provide a resolution function
    # normalized to a total area of 1 as our input.
    x = next_xp if next_xp > next_xin else next_xin
    # printf("  point xo:%g dx:%g x:%g, p:%ld, in:%ld, xp:%g, xin:%g\n",
    #       xo,dx,x,p,in,next_xp,next_xin);
    while True:
        # Step xin if we are at the next theory point
        if next_xin <= x:
            _in += 1
            if _in >= Nin:
                break  # At the right edge of the data
            next_xin = xin[_in]
            m1 = (yin[_in] - yin[_in - 1]) / (xin[_in] - xin[_in - 1])
            b1 = yin[_in] - m1 * xin[_in]

        # Step xp if we are at the next resolution point
        if next_xp <= x:
            p += 1
            if p >= Np:
                break  # At the right edge of the resolution
            next_xp = xo + dx * xp[p]
            m2 = (yp[p] - yp[p - 1]) / (xp[p] - xp[p - 1]) / dx
            b2 = yp[p] - m2 * next_xp

        # Find the next node
        next_x = next_xin if next_xin < next_xp else next_xp
        # Compute the convolution and norm between the current and next node
        delta = next_x - x
        delta2 = next_x * next_x - x * x
        delta3 = next_x * next_x * next_x - x * x * x
        norm += 0.5 * m2 * delta2 + b2 * delta
        sum += m1 * m2 / 3.0 * delta3 + 0.5 * (m1 * b2 + m2 * b1) * delta2 + b1 * b2 * delta
        # printf("  delta:%g delta2:%g delta3:%g norm:%g sum:%g m1:%g b1:%g m2:%g b2:%g x:%g nx:%g ni:%g np:%g\n",
        #  delta, delta2, delta3, norm, sum, m1, b1, m2, b2, x, next_x, next_xin, next_xp);
        # Move to the next node
        x = next_x

    return sum / norm


def convolve_sampled(xin, yin, xp, yp, x, dx, y):
    Nin = len(xin)
    Np = len(xp)
    N = len(x)

    # /* FIXME fails if xin are not sorted
    # * slow if x not sorted */
    if not Nin > 1:
        raise ValueError("Nin must be > 1")

    # /* Scan through all x values to be calculated */
    # /* Re: omp, each thread is going through the entire input array,
    # * independently, computing the resolution from the neighbourhood
    # * around its individual output points.  The firstprivate( in )
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

    _in = 0

    # ifdef _OPENMP
    # pragma omp parallel for firstprivate(in) schedule(static,1)
    # endif
    for _out in range(N):
        # /* width of resolution window for x is w = 2 dx ^ 2. */
        limit = -dx[_out] * xp[0]
        xo = x[_out]

        # /* Line up the left edge of the convolution window */
        # /* It is probably forward from the current position, */
        # /* but if the next dx is a lot higher than the current */
        # /* dx or if the x are not sorted, then it may be before */
        # /* the current position. * /
        # /* FIXME verify that the convolution window is just right */
        while _in < Nin - 1 and xin[_in] < xo - limit:
            _in += 1
        while _in > 0 and xin[_in] > xo - limit:
            _in -= 1

        # /* Special handling to avoid 0/0 for w = 0. */
        if dx[_out] > 0.0:
            # printf("convolve in:%ld out:%ld, xo:%g dx:%g\n",in,out,xo,dx[out])
            y[_out] = convolve_point_sampled(Nin, xin, yin, Np, xp, yp, xo, dx[_out], _in)
        elif _in < Nin - 1:
            # /* Linear interpolation */
            m = (yin[_in + 1] - yin[_in]) / (xin[_in + 1] - xin[_in])
            b = yin[_in] - m * xin[_in]
            y[_out] = m * xo + b
        elif _in > 0:
            # /* Linear extrapolation */
            m = (yin[_in] - yin[_in - 1]) / (xin[_in] - xin[_in - 1])
            b = yin[_in] - m * xin[_in]
            y[_out] = m * xo + b
        else:
            # /* Can't happen because there is more than one point in xin. */
            if not Nin > 1:
                raise ValueError("should be more than one point in Nin")
