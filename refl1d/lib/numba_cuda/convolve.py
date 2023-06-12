import numba
import numpy as np
from .clone_module import clone_module

MODULE = clone_module('refl1d.lib.python.convolve')

MODULE.prange = numba.prange

convolve_uniform = numba.njit('(f8[:], f8[:], f8[:], f8[:], f8[:])', cache=True, parallel=False)(MODULE.convolve_uniform)

convolve_gaussian_point = numba.njit('f8(f8[:], f8[:], i8, i8, f8, f8, f8)', cache=True, parallel=False, locals={
    "z": numba.float64,
    "Glo": numba.float64,
    "erflo": numba.float64,
    "erfmin": numba.float64,
    "y": numba.float64,
    "zhi": numba.float64,
    "Ghi": numba.float64,
    "erfhi": numba.float64,
    "m": numba.float64,
    "b": numba.float64,
})(MODULE.convolve_gaussian_point)
MODULE.convolve_gaussian_point = convolve_gaussian_point

# has same performance when using guvectorize instead of njit:
# @numba.guvectorize("(i8, f8[:], f8[:], i8, f8[:], f8[:], f8[:])", '(),(m),(m),(),(n),(n)->(n)')

convolve_gaussian = numba.njit("(f8[:], f8[:], f8[:], f8[:], f8[:])", cache=True, parallel=False, locals={
    "sigma": numba.float64,
    "xo": numba.float64,
    "limit": numba.float64,
    "k_in": numba.int64,
    "k_out": numba.int64,
})(MODULE.convolve_gaussian)


from math import erf, sqrt, exp, ceil, log2

PI4 = 12.56637061435917295385
PI_180 = 0.01745329251994329576
LN256 = 5.54517744447956247533
SQRT2 = numba.float32(1.41421356237309504880)
SQRT2PI = numba.float32(2.50662827463100050241)
LOG_RESLIMIT = -6.90775527898213703123
SQRT_NEG2_LOG_RESLIMIT = numba.float32(sqrt(-2 * LOG_RESLIMIT))
root_12_over_2 = sqrt(3)

# cuda implementation:
from numba import cuda

@cuda.jit
def convolve_gaussian_cuda_kernel(xin, yin, x, dx, y):
    # size_t in,out;
    Nin = numba.int32(len(xin))
    Nout = numba.int32(len(x))
    
    k_out = cuda.grid(1)
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
    k_in = numba.int32(0)
    
    if k_out < Nout:
        # /* width of resolution window for x is w = 2 dx^2. */
        NIN_MINUS_1 = numba.int32(Nin - numba.int32(1))
        sigma = dx[k_out]
        xo = x[k_out]
        limit = sigma * SQRT_NEG2_LOG_RESLIMIT
        XO_LOWER_LIMIT = numba.float32(xo - limit)

        # // if (out%20==0)

        # /* Line up the left edge of the convolution window */
        # /* It is probably forward from the current position, */
        # /* but if the next dx is a lot higher than the current */
        # /* dx or if the x are not sorted, then it may be before */
        # /* the current position. */
        # /* FIXME verify that the convolution window is just right */
        xx = xin[k_in]
        prev_xx = xx
        while (k_in < NIN_MINUS_1 and xx < XO_LOWER_LIMIT):
            k_in += numba.int32(1)
            prev_xx = xx
            xx = xin[k_in]
        while (k_in > 0 and xx > XO_LOWER_LIMIT):
            k_in -= numba.int32(1)
            prev_xx = xx
            xx = xin[k_in]

        # /* Special handling to avoid 0/0 for w=0. */
        if (sigma > numba.float32(0.)):
            two_sigma_sq = numba.float32(2.) * sigma * sigma
            # double z, Glo, erflo, erfmin, y

            z = xo - xx
            Glo = cuda.libdevice.expf(-z*z/two_sigma_sq)
            erfmin = erflo = cuda.libdevice.erff(-z/(SQRT2*sigma))
            yy = numba.float32(0.)
            while (k_in < NIN_MINUS_1):
                k_in += numba.int32(1)
                prev_xx = xx
                xx = xin[k_in]
                if (xx != prev_xx):
                    # /* No additional contribution from duplicate points. */

                    # /* Compute the next endpoint */
                    zhi = xo - xx
                    Ghi = cuda.libdevice.expf(-zhi*zhi/two_sigma_sq)
                    erfhi = cuda.libdevice.erff(-zhi/(SQRT2*sigma))
                    m = (yin[k_in]-yin[k_in-1])/(xx - prev_xx)
                    b = yin[k_in] - m * xx

                    # /* Add the integrals. */
                    yy += numba.float32(0.5)*(m*xo+b)*(erfhi-erflo) - sigma/SQRT2PI*m*(Ghi-Glo)

                    # /* Debug computation failures. */
                    # if isnan(y) {
                    #     print("NaN from %d: zhi=%g, Ghi=%g, erfhi=%g, m=%g, b=%g\n",
                    #          % (k,zhi,Ghi,erfhi,m,b))
                    # }

                    # /* Save the endpoint for next trapezoid. */
                    Glo = Ghi
                    erflo = erfhi

                    # /* Check if we've calculated far enough */
                    if (xx >= xo+limit):
                        break
                        
            y[k_out] = 2 * yy / (erflo - erfmin)

            # y[k_out] = convolve_gaussian_point(
            #     xin, yin, k_in, Nin, xo, limit, sigma)
        elif (k_in < Nin-numba.int32(1)):
            # /* Linear interpolation */
            m = (yin[k_in+numba.int32(1)]-yin[k_in])/(xin[k_in+numba.int32(1)]-xin[k_in])
            b = yin[k_in] - m*xin[k_in]
            y[k_out] = m*xo + b
        elif (k_in > numba.int32(0)):
            # /* Linear extrapolation */
            m = (yin[k_in]-yin[k_in-numba.int32(1)])/(xin[k_in]-xin[k_in-numba.int32(1)])
            b = yin[k_in] - m*xin[k_in]
            y[k_out] = m*xo + b
        else:
            # /* Can't happen because there is more than one point in xin. */
            # assert(Nin>1)
            pass


CACHESIZE = 64
DEBUG = False

@cuda.jit
def convolve_gaussian_cuda_local(xin, yin, x, dx, y):
    # size_t in,out;
    Nin = len(xin)
    Nout = len(x)
    
    # k_out = cuda.grid(1)
    
    block_x = cuda.blockIdx.x
    dim_x = cuda.blockDim.x
    tx = cuda.threadIdx.x
    sx = tx

    pos_x = tx + block_x * dim_x
    k_out = pos_x
    xin_offset = 0

    xb = cuda.shared.array(shape=(CACHESIZE,), dtype=numba.float32)
    yb = cuda.shared.array(shape=(CACHESIZE,), dtype=numba.float32)

    all_done = cuda.shared.array(shape=(1,), dtype=numba.boolean)
    all_done[0] = False # for all threads

    done = False # for this thread
    started = False

    integrated_result = 0.0
    interpolated_result = 0.0
    erfmin = 0.0
    erflo = 1.0
    
    if k_out < Nout:
        # /* width of resolution window for x is w = 2 dx^2. */
        sigma = dx[k_out]
        two_sigma_sq = 2. * sigma * sigma
        xo = x[k_out]
        limit = sqrt(-2.*sigma*sigma * LOG_RESLIMIT)

    while xin_offset < Nin and not all_done[0]:
        # fill the cache with a stride of block_x
        sx = tx
        lookup = sx + xin_offset
        if DEBUG and pos_x == 10:
            print('starting cache fill: ', xin_offset, sx, lookup, xin[lookup])
        while sx < CACHESIZE and lookup < Nin:
            xb[sx] = xin[lookup]
            yb[sx] = yin[lookup]
            sx += dim_x
            lookup = sx + xin_offset

        # wait for data to be loaded into shared buffers:
        cuda.syncthreads()
        if DEBUG and pos_x == 10:
            print('xb:', xb)        # print(xb)

        k_in = 0

        if k_out < Nout:
            # /* Line up the left edge of the convolution window */
            # /* It is probably forward from the current position, */
            # /* but if the next dx is a lot higher than the current */
            # /* dx or if the x are not sorted, then it may be before */
            # /* the current position. */
            # /* FIXME verify that the convolution window is just right */

            while (k_in < CACHESIZE-1 and ((k_in + xin_offset) < Nin-1) and xb[k_in] < xo-limit):
                k_in += 1
            while (k_in > 0 and xb[k_in] > xo-limit):
                k_in -= 1

            # double z, Glo, erflo, erfmin, y
            z = xo - xb[k_in]
            if k_in < CACHESIZE-1 and xb[k_in+1] >= xo - limit and not started:
                Glo = exp(-z*z/two_sigma_sq)
                erfmin = erflo = erf(-z/(SQRT2*sigma))
                started = True
                if DEBUG:
                    print('starting', z, xo, k_in, xb[k_in], tx, pos_x)

            while (started and not done and k_in < CACHESIZE-1):
                k_in += 1
                if DEBUG and pos_x == 10:
                    print('in loop:', pos_x, k_in, xb[k_in], xo+limit)
                if (xb[k_in] != xb[k_in-1]):
                    # /* No additional contribution from duplicate points. */

                    # /* Compute the next endpoint */
                    zhi = xo - xb[k_in]
                    Ghi = exp(-zhi*zhi/two_sigma_sq)
                    erfhi = erf(-zhi/(SQRT2*sigma))
                    m = (yb[k_in]-yb[k_in-1])/(xb[k_in]-xb[k_in-1])
                    b = yb[k_in] - m * xb[k_in]

                    # /* Add the integrals. */
                    integrated_result += 0.5*(m*xo+b)*(erfhi-erflo) - sigma/SQRT2PI*m*(Ghi-Glo)

                    # /* Debug computation failures. */
                    # if isnan(y) {
                    #     print("NaN from %d: zhi=%g, Ghi=%g, erfhi=%g, m=%g, b=%g\n",
                    #          % (k,zhi,Ghi,erfhi,m,b))
                    # }

                    # /* Save the endpoint for next trapezoid. */
                    Glo = Ghi
                    erflo = erfhi

                    # /* Check if we've calculated far enough */
                    if (xb[k_in] >= xo+limit):
                        done = True
                        if DEBUG:
                            print('done', pos_x)

        xin_offset += CACHESIZE
       
        if tx == 0:
            all_done[0] = done
        else:
            all_done[0] = all_done[0] and done

    # only one of integrated_result, interpolated_result will be non-zero:
    if k_out < Nout:
        y[k_out] = (2 * integrated_result / (erflo - erfmin)) + interpolated_result

def convolve_gaussian_cuda(xi, yi, x, dx, y):
    # import time
    # transfer_start = time.time()
    xi_d = cuda.to_device(xi.astype(np.float32))
    yi_d = cuda.to_device(yi.astype(np.float32))
    x_d = cuda.to_device(x.astype(np.float32))
    dx_d = cuda.to_device(dx.astype(np.float32))
    y_d = cuda.to_device(y.astype(np.float32))
    # transfer_end = time.time()
    # print(f"transfer to gpu time: {transfer_end - transfer_start}")
    
    threadsperblock = 32
    blockspergrid = (y.shape[0] + (threadsperblock - 1)) // threadsperblock
    # print(blockspergrid)
    
    convolve_gaussian_cuda_kernel[blockspergrid, threadsperblock](xi_d, yi_d, x_d, dx_d, y_d)
    # convolve_gaussian_cuda_newton[blockspergrid, threadsperblock](xi_d, yi_d, x_d, dx_d, y_d)
    # convolve_gaussian_cuda_local[blockspergrid, threadsperblock](xi_d, yi_d, x_d, dx_d, y_d)
    # calc_end = time.time()
    # print(f"calc time: {calc_end - transfer_end}")
    y[:] = y_d.copy_to_host()[:]
    # copy_end = time.time()
    # print(f"copy time: {copy_end - calc_end}")
