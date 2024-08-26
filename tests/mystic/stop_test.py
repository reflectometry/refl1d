# This program is public domain
import sys

sys.path.append("..")

import numpy as np
import mystic.termination as stop
from mystic.history import History


def _check(success=None, failure=None):
    """
    Create an artificial history and run the tests
    """

    history = History(step=0, calls=0, time=0, cpu_time=0, point=0, value=0, population_points=0, population_values=0)
    # print success,failure
    for t in success.primitives() | failure.primitives():
        t.config_history(history)

    history.ndim = 2
    history.lower_bound = np.array([-1, 0], "d")
    history.upper_bound = np.array([1, 1], "d")
    history.update(
        step=1,
        calls=2,
        time=6,
        cpu_time=10,
        point=np.array([1, 1], "d"),
        value=75,
        population_points=np.array([[0, 0], [1, 1]], "d"),
        population_values=np.array([100, 75], "d"),
    )
    # How to make this cumulative?
    history.update(
        step=2,
        calls=4,
        time=11,
        cpu_time=18,
        point=np.array([1, 0.5]),
        value=50,
        population_points=np.array([[0.5, 0], [1, 0.5]], "d"),
        population_values=np.array([87, 50], "d"),
    )

    return success.status(history), failure.status(history)


def test():
    # TODO need to test all conditions, scaled and unscaled

    # Check that tests succeed when expected
    success = stop.Df(0.5) & stop.Dx(0.8)
    failure = stop.Steps(2) & stop.Calls(3)
    # print success
    # print failure
    s, f = _check(success=success, failure=failure)
    # print s[0],", ".join(str(c) for c in s[1])
    # print f[0],", ".join(str(c) for c in f[1])
    assert s[0], str(success)
    assert f[0], str(failure)
    assert set(s[1]) == success.primitives()
    assert set(f[1]) == failure.primitives()

    # Check that tests fail when expected
    success = stop.Df(0.05) & stop.Dx(0.08)
    failure = stop.Steps(4) & stop.Calls(6)
    s, f = _check(success=success, failure=failure)
    # print s[0],", ".join(str(c) for c in s[1])
    # print f[0],", ".join(str(c) for c in f[1])
    assert not s[0], str(~success)
    assert not f[0], str(~failure)
    # We know all conditions should be negated on failure
    assert set(c.condition for c in s[1]) == success.primitives()
    assert set(c.condition for c in f[1]) == failure.primitives()
    print("need to test all termination conditions")


if __name__ == "__main__":
    import doctest

    doctest.testmod(stop)
    test()
