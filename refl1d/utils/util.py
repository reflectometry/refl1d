__all__ = ["merge_ends"]

import numpy as np


def asbytes(s):
    return s.encode("utf-8")


def merge_ends(w, p, tol=1e-3):
    """
    join the leading and trailing ends of the profile together so fewer
    slabs are required and so that gaussian roughness can be used.
    """
    # TODO: accept rho, rhoi pairs as well
    # TODO: make sure we apply an interface to the right as well as the left
    try:
        # Assuming there p[0] != p[-1] within tolerance, we are guaranteed
        # that we will have a first value not equivalent to p[0] on the
        # left, with index > 0 and a last value not equivalent to p[-1] on
        # the right, with index < -1.  We are going to put the first value
        # at left index - 1 and the last value at right index + 1, accumulating
        # the widths of the identical layers.
        lidx = np.where(abs(p - p[0]) > tol)[0][0] - 1
        ridx = len(p) - np.where(abs(p[::-1] - p[-1]) > tol)[0][0]
        w[lidx], p[lidx] = np.sum(w[: lidx + 1]), p[0]
        w[ridx], p[ridx] = np.sum(w[ridx:]), p[-1]
        return w[lidx : ridx + 1], p[lidx : ridx + 1]
    except Exception:
        if len(w):
            # All one big layer
            w[0] = np.sum(w)
            return w[0:1], p[0:1]
        else:
            return w, p
