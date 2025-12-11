# This program is public domain
import sys

sys.path.append("..")
import mystic.condition as condition
from mystic.condition import true, false, Constant, Not


def test():
    assert true()
    assert not false()

    assert not (~true)()
    assert (~false)()

    assert (true & true)()
    assert not (true & false)()
    assert not (false & true)()
    assert not (false & false)()

    assert (true | true)()
    assert (true | false)()
    assert (false | true)()
    assert not (false | false)()

    assert (true ^ false)()
    assert (false ^ true)()
    assert not (false ^ false)()
    assert not (true ^ true)()

    assert ((true | false) & true)()
    assert not (~(true | false) & true)()

    false2 = Constant(False)
    true2 = Constant(True)
    assert true.status() == (True, [true])
    assert false.status() == (False, [Not(false)])

    assert (~true).status() == (False, [true])
    assert (~false).status() == (True, [Not(false)])

    assert (true & true2).status() == (True, [true, true2])
    assert (true & false).status() == (False, [Not(false)])
    assert (false & true).status() == (False, [Not(false)])
    assert (false & false2).status() == (False, [Not(false), Not(false2)])

    assert (true | true2).status() == (True, [true, true2])
    assert (true | false).status() == (True, [true])
    assert (false | true).status() == (True, [true])
    assert (false | false2).status() == (False, [Not(false), Not(false2)])

    assert (true ^ false).status() == (True, [true, Not(false)])
    assert (false ^ true).status() == (True, [Not(false), true])
    assert (true ^ true2).status() == (False, [true, true2])
    assert (false ^ false2).status() == (False, [Not(false), Not(false2)])

    assert ((true | false) & true2).status() == (True, [true, true2])
    assert (~(true2 | false) & true).status() == (False, [true2])

    assert set((~(true2 | false) & true).primitives()) == set([true, false, true2])


if __name__ == "__main__":
    import doctest

    doctest.testmod(condition)
    test()
