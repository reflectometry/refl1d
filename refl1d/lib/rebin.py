import numba
import numba.experimental
import math

# Define a bin iterator to adapt to either forward or reversed inputs.
spec = [
    ('forward', numba.boolean),
    ('n', numba.int64),
    ('edges', numba.float64[:]),
    ('bin', numba.int64),
    ('lo', numba.float64),
    ('hi', numba.float64),
    ('atend', numba.boolean)
]


@numba.experimental.jitclass(spec)
class BinIter:
    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        self.forward = edges[0] < edges[n]

        if (self.forward):
            self.bin = 0
            self.lo = edges[0]
            self.hi = edges[1]
        else:
            self.bin = n - 1
            self.lo = edges[n]
            self.hi = edges[n-1]

        self.atend = n < 1

    def increment(self):
        if self.atend:
            raise IndexError("moving beyond final bin")
        self.lo = self.hi
        if self.forward:
            self.bin += 1
            self.atend = (self.bin >= self.n)
            if (not self.atend):
                self.hi = self.edges[self.bin+1]
        else:
            self.atend = (self.bin == 0)
            if not self.atend:
                self.bin -= 1
                self.hi = self.edges[self.bin]
        return self


REBIN_COUNTS_PORTION_SIGS = [
    "void(i8, f8[:], i4[:], i8, f8[:], i4[:], f8)",
    "void(i8, f8[:], f8[:], i8, f8[:], f8[:], f8)",
]


@numba.njit(REBIN_COUNTS_PORTION_SIGS, parallel=False, cache=True)
def rebin_counts_portion(Nold, vold, Iold, Nnew, vnew, Inew, ND_portion):

    # Note: inspired by rebin from OpenGenie, but using counts per bin
    # rather than rates.

    # Does not work in place
    #assert(Iold != Inew)

    # Traverse both sets of bin edges; if there is an overlap, add the portion
    # of the overlapping old bin to the new bin.
    _from = BinIter(Nold, vold)
    _to = BinIter(Nnew, vnew)
    while (not _from.atend and not _to.atend):
        # std::cout << "from " << from.bin << ": [" << from.lo << ", " << from.hi << "]\n";
        # std::cout << "to " << to.bin << ": [" << to.lo << ", " << to.hi << "]\n";
        if (_to.hi <= _from.lo):
            _to.increment()  # new must catch up to old
        elif (_from.hi <= _to.lo):
            _from.increment()  # old must catch up to new
        else:
            overlap = min(_from.hi, _to.hi) - max(_from.lo, _to.lo)
            portion = overlap/(_from.hi - _from.lo)
            Inew[_to.bin] += round(Iold[_from.bin]*portion*ND_portion)
            if (_to.hi > _from.hi):
                _from.increment()
            else:
                _to.increment()


REBIN_COUNTS_SIGS = [
    "void(i8, f8[:], i4[:], i8, f8[:], i4[:])",
    "void(i8, f8[:], f8[:], i8, f8[:], f8[:])",
]


@numba.njit(REBIN_COUNTS_SIGS, parallel=False, cache=True)
def rebin_counts_old(Nold, xold, Iold, Nnew, xnew, Inew):

    # Note: inspired by rebin from OpenGenie, but using counts per bin
    # rather than rates.

    # Clear the new bins
    for i in range(Nnew):
        Inew[i] = 0

    rebin_counts_portion(Nold, xold, Iold, Nnew, xnew, Inew, 1.)


REBIN_COUNTS_SIGS = [
    "void(f8[:], i4[:], f8[:], i4[:])",
    "void(f8[:], f8[:], f8[:], f8[:])",
]


@numba.njit(REBIN_COUNTS_SIGS, parallel=False, cache=True)
def rebin_counts(xold, Iold, xnew, Inew):

    # Note: inspired by rebin from OpenGenie, but using counts per bin
    # rather than rates.
    Nold = len(Iold)
    Nnew = len(Inew)
    # Clear the new bins
    for i in range(Nnew):
        Inew[i] = 0

    rebin_counts_portion(Nold, xold, Iold, Nnew, xnew, Inew, 1.)


REBIN_INTENSITY_SIG = "void(i4, f8[:], f8[:], f8[:], i4, f8[:], f8[:], f8[:])"


@numba.njit(REBIN_INTENSITY_SIG, parallel=False, cache=True)
def rebin_intensity(Nold, xold, Iold, dIold, Nnew, xnew, Inew, dInew):

    # Note: inspired by rebin from OpenGenie, but using counts per bin rather than rates.

    # Clear the new bins
    for i in range(Nnew):
        dInew[i] = Inew[i] = 0

    # Traverse both sets of bin edges; if there is an overlap, add the portion
    # of the overlapping old bin to the new bin.
    _from = BinIter(Nold, xold)
    _to = BinIter(Nnew, xnew)
    while (not _from.atend and not _to.atend):
        # std::cout << "from " << from.bin << ": [" << from.lo << ", " << from.hi << "]\n";
        # std::cout << "to " << to.bin << ": [" << to.lo << ", " << to.hi << "]\n";
        if (_to.hi <= _from.lo):
            _to.increment()  # new must catch up to old
        elif (_from.hi <= _to.lo):
            _from.increment()  # old must catch up to new
        else:
            overlap = min(_from.hi, _to.hi) - max(_from.lo, _to.lo)
            portion = overlap/(_from.hi - _from.lo)

            Inew[_to.bin] += round((Iold[_from.bin])*portion)
            # add in quadrature
            dInew[_to.bin] += round((dIold[_from.bin]*portion)**2)
            if (_to.hi > _from.hi):
                _from.increment()
            else:
                _to.increment()

    # Convert variance to standard deviation.
    for i in range(Nnew):
        dInew[i] = math.sqrt(dInew[i])
