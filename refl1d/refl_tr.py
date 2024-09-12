# This program is public domain.
r"""
Optical matrix form of the reflectivity calculation.

This is a pure python implementation of reflectometry which returns the
reflection and transmission at each layer.  Based on C-D matrix form
given in [1].

***WARNING*** Returns the conjugate of the reflectometry amplitude compared
to the abeles matrix calculation given on wikipedia.[2]

***WARNING*** Calculation not yet trusted: $r^2 + t^2 \ne 1$ for perfect
interfaces with no absorption.  However, $r^2 + t^2 k0/kn = 1$.  This detail
comes from the tmm package transmission power function.[3]

***WARNING*** Transmission value explodes below the critical edge.

***WARNING*** Nevot-Croce does not preserve $r^2 + t^2 k0/kn = 1$

Note: for conversion to GPU with parallel prefix matrix multiplication[4],
need to restructure the calculation so the matrices are independent, and
we don't need to first perform the cumulative sum on depth z.  The Abeles
matrix algorithm[2] does this, but it is not computing transmissions
correctly.

[1] Dura, J.A., Rus, E.D., Kienzle, P.A., Maranville, B.B., 2017.
Nanolayer Analysis by Neutron Reflectometry,
in: Imae, T. (Ed.), Nanolayer Research. Elsevier, Amsterdam, pp. 155-202.
https://doi.org/10.1016/B978-0-444-63739-0.00005-0

[2] https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)#Abeles_matrix_formalism

[3] https://github.com/sbyrnes321/tmm/blob/4ebbe9dbee140ccabad22c158107681846c9d204/tmm_core.py#L161-L181

[4] https://en.wikipedia.org/wiki/Prefix_sum
"""
from __future__ import print_function, division

import numpy as np
from numpy import asarray, isscalar, empty, ones, ones_like
from numpy import sqrt, exp, pi

def refl_tr(kz, depth, rho, irho=0, sigma=0, rho_index=None):
    r"""
    Reflectometry as a function of kz for a set of slabs.

    :Parameters:

    *kz* : float[n] | |1/Ang|
        Scattering vector $2\pi\sin(\theta)/\lambda$. This is $\tfrac12 Q_z$.
    *depth* :  float[m] | |Ang|
        thickness of each layer.  The thickness of the incident medium
        and substrate are ignored.
    *rho*, *irho* :  float[n, k] | |1e-6/Ang^2|
        real and imaginary scattering length density for each layer for each kz
        Note: absorption cross section mu = 2 irho/lambda
    *sigma* : float[m-1] | |Ang|
        interfacial roughness.  This is the roughness between a layer
        and the subsequent layer.  There is no interface associated
        with the substrate.  The sigma array should have at least m-1
        entries, though it may have m with the last entry ignored.
    *rho_index* : int[m]
        index into rho vector for each kz

    Returns array of shape [k X 2 X n], with R[k, 0, n] as the transmission
    amplitude and R[k, 1, n] as the reflection amplitude at the layer k
    interface for kz[n].

    Slabs are ordered with the incident SLD at index 0 and backing at
    index -1.
    """
    if isscalar(kz):
        kz = asarray([kz], 'd')

    # Don't support kz negative for now
    if (kz < 0).any():
        raise ValueError("require kz >= 0")

    m = len(depth)

    # Make everything into arrays
    depth = asarray(depth, 'd')
    rho = asarray(rho, 'd')
    irho = irho*ones_like(rho) if isscalar(irho) else asarray(irho, 'd')
    sigma = sigma*ones(m-1, 'd') if isscalar(sigma) else asarray(sigma, 'd')

    # Repeat rho, irho columns as needed
    if rho_index is not None:
        rho = rho[rho_index, :]
        irho = irho[rho_index, :]
    elif len(rho.shape) == 1:
        rho = rho[None, :]
        irho = irho[None, :]

    # Force the correct branch cut for sqrt below the critical edge
    irho = abs(irho) + 1e-30

    r = _calc(kz, depth, rho, irho, sigma)
    return r


def _calc(kz, depth, rho, irho, sigma):
    # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray

    num_kz = len(kz)
    num_interfaces = len(depth) - 1

    # Complex index of refraction is relative to the incident medium.
    # Dura, J.A., Rus, E.D., Kienzle, P.A., Maranville, B.B., 2017.
    # Nanolayer Analysis by Neutron Reflectometry,
    # in: Imae, T. (Ed.), Nanolayer Research. Elsevier, Amsterdam, pp. 155-202.
    # https://doi.org/10.1016/B978-0-444-63739-0.00005-0
    #
    # Neutron energy in the layer, E0, is:
    #     E0 = h_bar^2/(2 m_n) (k_l^2 + 4 pi rho_l)
    # For the incident neutron, E0 is defined by k_z and incident rho:
    #     E0 = h_bar^2/(2 m_n) (k_z^2 + 4 pi rho_in)
    # Solving for k_l:
    #     k_l = sqrt((2 m_n)/h_bar E0 - 4 pi rho_l)
    # Substituting E0:
    #     k_l = sqrt(k_z^2 + 4 pi rho_in - 4 pi rho_l)
    #rho_in = np.where(kz >= 0, rho[:, 0], rho[:, -1])
    rho_in = rho[:, 0]
    kz_sq = kz*kz

    k = sqrt(kz_sq + 4e-6*pi*(rho_in - rho[:, 0]))
    z = 0
    M = []
    k_list = [k]
    B11 = 1
    B22 = 1
    B21 = 0
    B12 = 0
    for i in range(num_interfaces):
        k_next = sqrt(kz_sq + 4e-6*pi*(rho_in - rho[:, i+1] + 1j*irho[:, i+1]))
        nevot_croce_damping = exp(-2*k*k_next*sigma[i]**2)
        k_plus, k_minus = k_next + k, k_next - k
        M11 = k_plus * exp(-1j*k_minus*z)/(2*k)
        M22 = k_plus * exp(+1j*k_minus*z)/(2*k)
        M12 = nevot_croce_damping * k_minus * exp(-1j*k_plus*z)/(2*k)
        M21 = nevot_croce_damping * k_minus * exp(+1j*k_plus*z)/(2*k)
        M.append((M11, M12, M21, M22))
        k_list.append(k_next)
        # Right-multiply: B = M*B
        C11 = M11*B11 + M12*B21
        C21 = M21*B11 + M22*B21
        B11, B21 = C11, C21
        C12 = M11*B12 + M12*B22
        C22 = M21*B12 + M22*B22
        B12, B22 = C12, C22
        k = k_next
        z += depth[i+1]
        #print("==== layer %d ====" % i)
        #print("B11:", B11)
        #print("B22:", B22)
        #print("B21:", B21)
        #print("B12:", B12)

    r = -B21/B22
    t = B11 + B12*r

    results = np.empty((num_interfaces+1, 2, num_kz), 'D')
    if 1:
        # Propagate (1, r) forward using:
        #    [c_{n+1}, d_{n+1}]^T = M_n [c_n, d_n]^T
        #print("propagate forward")
        c, d = np.ones(num_kz, 'D'), r
        results[0] = c, d
        for i in range(num_interfaces):
            M11, M12, M21, M22 = M[i]
            c, d = M11*c + M12*d, M21*c + M22*d
            results[i+1] = c, d
    else:
        # Propagate (t, 0) backward using:
        #    [c_n, d_n]^T = M_n^{-1} [c_{n+1}, d_{n+1}]^T
        #print("propagate backward")
        c, d = t, np.zeros(num_kz, 'D')
        results[-1] = c, d
        for i in reversed(range(num_interfaces)):
            M11, M12, M21, M22 = M[i]
            # Minv = [[M22, -M12], [-M21, M11]] / (M11 M22 - M12 M21)
            det = B11*B22 - B21*B12
            c, d = (M22*c - M12*d)/det, (-M21*c + M11*d)/det
            results[i] = c, d

    #print("r   ", r)
    #for k, layer in enumerate(results):
    #    print("r[%d]"%k, layer[1])
    #    print("t[%d]"%k, layer[0])
    #print("t   ", t)
    k0, kn = k_list[0], k_list[-1]
    #print("r^2 =", abs(r**2))
    #print("t^2 =", abs(t**2))
    #print("r^2 + t^2*k0/kn =", abs(r**2) + abs(t**2)*(k0/kn).real)
    #print("k0", k0)
    #print("kn", kn)

    ## Compute back reflectivity
    #negative = (kz < 0.0)
    #if negative.any():
    #    det = B11*B22 - B21*B12
    #    r[negative] = -(B12 / B22 / det)[negative]
    return results

def check():
    import numpy as np
    np.set_printoptions(linewidth=10000)

    #q = np.linspace(-0.3, 0.3, 4)
    q = np.linspace(0.1, 0.3, 3)
    layers = [
        # depth rho irho sigma
        [  0, 0.0, 0.0,  0.0],
        [200, 2.0, 0.0,  0.0],
        [200, 4.0, 0.0,  0.0],
        [  0, 3.0, 0.0,  0.0],
    ]
    #layers[1][2] = 1.0  # add absorption to layer 1
    #for L in layers[:-1]: L[3] = 10.0   # add roughness to all layers

    depth, rho, irho, sigma = (np.asarray(v) for v in zip(*layers))
    rho *= 100  # show point below critical edge
    print("q", q)
    try:
        from .abeles import refl
        r_old = refl(q/2, depth, rho, irho=irho, sigma=sigma)
        print("rold", r_old)
    except ImportError:
        print("could not import abeles")
    wave = refl_tr(q/2, depth, rho, irho=irho, sigma=sigma)
    for k, wk in enumerate(wave):
        print("r[%d]"%k, wk[1])
        print("t[%d]"%k, wk[0])
    r, t = wk[0][1], wk[-1][0]
    #print("r^2", abs(r**2))


def check2(t=200, rho=4.66):
    import numpy as np
    np.set_printoptions(linewidth=10000)

    #q = np.linspace(0.0, 0.05, 600)
    q = np.linspace(0.0, 0.0005, 6000)
    #q = np.linspace(0.1, 0.3, 3)
    layers = [
        # depth rho irho sigma
        [  0, 0.0, 0.0,  0.0],
        #[200, 2.0, 0.0,  0.0],
        [t, -rho, 0.0,  0.0],
        [  0, 0, 0.0,  0.0],
        #[  0, 2.07, 0.0,  0.0],
    ]
    # add absorption
    # layers[1][2] = 1.0
    depth, rho, irho, sigma = zip(*layers)

    wave = refl_tr(q/2, depth, rho, irho=irho, sigma=sigma)
    #print("shape", wave.shape)
    r = wave[0, 1]

    import matplotlib.pyplot as plt
    plt.plot(q, r.real, '-b', label='real')
    plt.plot(q, r.imag, '-r', label='imag')
    plt.plot(q, abs(r**2), '-g', label='magnitude')
    #plt.legend()
    plt.grid(True)
    #plt.xlabel('q (1/A)')
    #plt.title("stack = Si / 20 nm Au / Air")
    #plt.show()
    #print("q", q)
    #print("r", r)
    #print("r^2", abs(r**2))

def check3(t=200, rho=4.66):
    import numpy as np
    np.set_printoptions(linewidth=10000)

    qmax = 0.1
    t, qmax = 2000, 0.05
    #t, qmax = 20, 0.4
    q = np.linspace(0.0, qmax, 6000)[1:]
    layers = [
        # depth rho irho sigma
        [  0, 0.0, 0.0,  0.0],
        #[200, 2.0, 0.0,  0.0],
        [t, rho, 0.0,  0.0],
        [  0, 0, 0.0,  0.0],
        #[  0, 2.07, 0.0,  0.0],
    ]
    # add absorption
    # layers[1][2] = 1.0
    vdepth, vrho, virho, vsigma = [np.asarray(v) for v in zip(*layers)]

    if 1:
        steps = 2001
        #portion = 0.9
        portion = 0.0
        thickness = t
        flat = thickness*portion
        ends = thickness*(1-portion)
        z = np.linspace(-3, 3, steps)
        vrho = np.exp(-0.5*z**2)*rho
        virho = 0*rho
        vdepth = np.ones_like(z)*ends/steps
        vdepth[int((steps-1)/2)] = flat
        vsigma = 0*rho

    if 0:
        steps = 2001
        thickness, portion = t, 0.0
        flat = thickness*portion
        ends = thickness*(1-portion)
        z = np.linspace(-3, 3, steps)
        vrho = np.exp(-0.5*z**2)*rho
        vrho += -np.exp(-0.5*(z-1)**2*4)*rho
        virho = 0*rho
        vdepth = np.ones_like(z)*ends/steps
        vdepth[int((steps-1)/2)] = flat
        vsigma = 0*rho

    ## normalize to same amount of scattering density
    #total = rho*t
    #vrho *= total/np.sum(vrho*vdepth)

    wave = refl_tr(q/2, vdepth, vrho, irho=virho, sigma=vsigma)
    #print("shape", layers.shape)
    r = wave[0, 1]

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid.inset_locator import InsetPosition

    scale = q*0+1
    #scale = q**4
    plt.plot(q, scale*r.real, '-b', label='real', zorder=2)
    plt.plot(q, scale*r.imag, '-r', label='imag')
    plt.plot(q, scale*abs(r**2), '-g', label='magnitude')
    #plt.yscale('symlog', linthreshy=1e-6)
    #plt.legend()
    plt.grid(True)
    ax = plt.gca()
    inset = plt.axes([0, 0, 1, 1])
    inset.step(np.cumsum(vdepth), vrho, '-g', label='rho')
    #inset.step(np.cumsum(vdepth), virho, '-m', label='irho')
    inset.set_axes_locator(InsetPosition(ax, [0.6, 0.7, 0.4, 0.3]))
    inset.patch.set_alpha(0.5)
    inset.patch.set_color([0.9,0.9,0.8])
    #mark_inset(ax, inset)

    #plt.xlabel('q (1/A)')
    #plt.title("stack = Si / 20 nm Au / Air")
    #plt.show()
    #print("q", q)
    #print("r", r)
    #print("r^2", abs(r**2))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    check3()
    """
    plt.subplot(211)
    check2(t=50)
    check2(t=100)
    check2(t=200)
    check2(t=400)
    check2(t=800)
    check2(t=1600)
    plt.subplot(212)
    check2(t=200,rho=-2)
    check2(t=200,rho=1)
    check2(t=200,rho=2)
    check2(t=200,rho=4)
    check2(t=200,rho=6)
    check2(t=200,rho=8)
    """
    plt.show()
