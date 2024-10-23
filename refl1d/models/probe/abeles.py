"""
Optical matrix form of the reflectivity calculation.

O.S. Heavens, Optical Properties of Thin Solid Films

This is a pure python implementation of reflectometry provided for
convenience when a compiler is not available.  The refl1d
application uses reflmodule to compute reflectivity.
"""

from numpy import asarray, isscalar, empty, ndarray, ones, ones_like
from numpy import sqrt, exp, pi


def refl(kz, depth, rho, irho=0, sigma=0, rho_index=None):
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

    Slabs are ordered with the surface SLD at index 0 and substrate at
    index -1, or reversed if kz < 0.
    """
    if isscalar(kz):
        kz = asarray([kz], "d")

    m = len(depth)

    # Make everything into arrays
    depth = asarray(depth, "d")
    rho = asarray(rho, "d")
    irho = irho * ones_like(rho) if isscalar(irho) else asarray(irho, "d")
    sigma = sigma * ones(m - 1, "d") if isscalar(sigma) else asarray(sigma, "d")

    # Repeat rho, irho columns as needed
    if rho_index is not None:
        rho = rho[rho_index, :]
        irho = irho[rho_index, :]
    elif len(rho.shape) == 1:
        rho = rho[None, :]
        irho = irho[None, :]

    # Force the correct branch cut for sqrt below the critical edge
    irho = abs(irho) + 1e-30

    ## For kz < 0 we need to reverse the order of the layers
    ## Note that the interface array sigma is conceptually one
    ## shorter than rho, mu so when reversing it, start at n-1.
    ## This allows the caller to provide an array of length n
    ## corresponding to rho, mu or of length n-1.
    r = empty(len(kz), "D")
    r[kz >= 1e-10] = _calc(kz[kz >= 1e-10], depth, rho, irho, sigma)
    r[kz <= 1e-10] = _calc(-kz[kz <= 1e-10], depth[::-1], rho[:, ::-1], irho[:, ::-1], sigma[m - 2 :: -1])
    r[abs(kz) < 1e-10] = -1
    return r


def _calc(kz: ndarray, depth: ndarray, rho: ndarray, irho: ndarray, sigma: ndarray) -> ndarray:
    """Reflectivity as a function of kz for a set of slabs."""
    if len(kz) == 0:
        return kz

    # Complex index of refraction is relative to the incident medium.
    # We can get the same effect using kz_rel^2 = kz^2 + 4*pi*rho_o
    # in place of kz^2, and ignoring rho_o
    kz_sq = kz**2 + 4e-6 * pi * rho[:, 0]
    k = kz

    # According to Heavens, the initial matrix should be [ 1 F; F 1],
    # which we do by setting B=I and M0 to [1 F; F 1].  An extra matrix
    # multiply versus some coding convenience.
    B11 = 1
    B22 = 1
    B21 = 0
    B12 = 0
    for i in range(0, len(depth) - 1):
        k_next = sqrt(kz_sq - 4e-6 * pi * (rho[:, i + 1] + 1j * irho[:, i + 1]))
        F = (k - k_next) / (k + k_next)
        F *= exp(-2 * k * k_next * sigma[i] ** 2)
        # print("==== layer", i)
        # print("kz:", kz.real)
        # print("k:", k.real)
        # print("k_next:", k_next.real)
        # print("F:", F.real)
        # print("rho:", rho[:, i+1])
        # print("irho:", irho[:, i+1])
        # print("d:", depth[i], "sigma:", sigma[i])
        M11 = exp(1j * k * depth[i]) if i > 0 else 1
        M22 = exp(-1j * k * depth[i]) if i > 0 else 1
        M21 = F * M11
        M12 = F * M22
        C1 = B11 * M11 + B21 * M12
        C2 = B11 * M21 + B21 * M22
        B11 = C1
        B21 = C2
        C1 = B12 * M11 + B22 * M12
        C2 = B12 * M21 + B22 * M22
        B12 = C1
        B22 = C2
        k = k_next
        # print("B11:", B11)
        # print("B22:", B22)
        # print("B21:", B21)
        # print("B12:", B12)
        # print("1-det:", 1 - (B11*B22 - B21*B12))

    r = B12 / B11
    return r


def test():
    import numpy as np

    np.set_printoptions(linewidth=10000)

    q = np.linspace(-0.3, 0.3, 6)
    # q = np.linspace(0.1, 0.3, 3)
    layers = [
        # depth rho irho sigma
        [0, 1.0, 0.0, 10.0],
        [200, 2.0, 1.0, 10.0],
        [200, 4.0, 0.0, 10.0],
        [0, 2.0, 0.0, 0.0],
    ]
    # add absorption
    # layers[1][2] = 1.0

    depth, rho, irho, sigma = zip(*layers)
    r = refl(q / 2, depth, rho, irho=irho, sigma=sigma)
    print("q", q)
    print("r", r)
    # print("r^2", abs(r**2))


if __name__ == "__main__":
    test()
