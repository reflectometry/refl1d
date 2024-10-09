import numpy as np

from refl1d.models.probe.abeles import refl


def test():
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
