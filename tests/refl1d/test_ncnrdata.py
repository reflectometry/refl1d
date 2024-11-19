import os

from refl1d.probe.data_loaders import ncnrdata

testdir = os.path.dirname(__file__)


def test():
    header, data = ncnrdata.parse_file(os.path.join(testdir, "cg1test.refl"))
    print(header)
    print(data)


if __name__ == "__main__":
    test()
