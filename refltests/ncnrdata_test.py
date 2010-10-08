import sys; sys.path.append('..')
import os
from refl1d import ncnrdata

testdir = os.path.dirname(__file__)
def test():
    header,data = ncnrdata.parse_file(os.path.join(testdir,'cg1test.refl'))
    print header
    print data

if __name__ == "__main__":
    test()
