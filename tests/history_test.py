# This program is in the public domain
import sys; sys.path.append("..")
import mystic.history as history

def test():
    h = history.History(spam=3)

    assert h.spam.keep == 3
    #print h.spam.keep
    h.provides(eggs=2)
    h.requires(spam=2, eggs=2)
    #print h.spam.keep
    assert h.spam.keep == 3
    assert h.eggs.keep == 2
    h.requires(spam=4)
    assert h.spam.keep == 4

    # Make sure the right things are in the queue
    for i in range(5): h.spam.put(i)
    assert len(h.spam) == 4
    assert h.spam[0] == 4
    assert h.spam[3] == 1

    # History is non-modifiable
    try:
        h.spam[0] = 3
    except TypeError:
        pass
    else:
        raise Exception("assignment should raise TypeError")

    try:
        v = h.spam[-1]
        raise Exception('access only from the front')
    except IndexError:
        pass

    try:
        v = h.spam[10]
        raise Exception('out of bounds')
    except IndexError:
        pass

    try:
        h.requires(beans=3)
        raise Exception('provide before requiring')
    except AttributeError:
        pass

    try:
        h.provides(spam=3)
        raise Exception('already provided')
    except AttributeError:
        pass

    try:
        h.provides(provides=0)
        raise Exception('pre-existing name')
    except AttributeError:
        pass

    try:
        h.requires(provides=2)
        raise Exception('history method as trace name')
    except AttributeError:
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod(history)
    test()
