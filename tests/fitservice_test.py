# This program is public domain
import sys; sys.path.append('..')
import mystic.fitservice as fit
from mystic.examples.simple import f

def test():
    result = fit.fit( (f,[2,4],None) )
    print "result",result

if __name__ == "__main__":
    import doctest
    #doctest.testmod(module)
    test()
