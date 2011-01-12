# This program is public domain
import sys; sys.path.insert(0, '..')
import mystic.bounds as bounds

def test_bounds(b,v,err):
    msg = "%s at %g"%(b,v)
    assert 0.5 <= b.get01(v) <= 1,msg
    assert 0 <= b.get01(-v) <= 0.5,msg
    rp = b.put01(b.get01(v))
    rm = b.put01(b.get01(-v))
    assert abs((rp-v)/v) < err,(msg,rp)
    assert abs((rm+v)/v) < err,(msg,rm)
def test_all_codecs():
    # Atan: test_codec(bounds.Unbounded(),0.825*2**20, 1e-10)
    test_bounds(bounds.Unbounded(), 1e-5, 1e-10)
    test_bounds(bounds.Unbounded(), 1e5, 1e-10)
    test_bounds(bounds.Unbounded(), 1e25, 1e-10)
    test_bounds(bounds.Unbounded(), 1e300, 1e-10)
    sp = bounds.BoundedBelow(1e5)
    assert sp.get01(1e5) < 1e-10
    assert abs(sp.put01(sp.get01(1e5))/1e5-1) < 1e-9
    assert abs(sp.put01(sp.get01(1e15))/1e15-1) < 1e-9
    assert abs(sp.put01(sp.get01(1e308))/1e308-1) < 1e-9
    sm = bounds.BoundedAbove(-1e5)
    assert abs(sm.get01(-1e5)-1) < 1e-10
    assert abs(sm.put01(sm.get01(-1e5))/-1e5-1) < 1e-9
    assert abs(sm.put01(sm.get01(-1e15))/-1e15-1) < 1e-9
    assert abs(sm.put01(sm.get01(-1e308))/-1e308-1) < 1e-9
    print "need more bounds tests"

def test():
    test_all_codecs()
    # TODO test the bounds

if __name__ == "__main__":
    import doctest
    doctest.testmod(bounds)
    test()
