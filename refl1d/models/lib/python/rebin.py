import math


class BinIter:
    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        self.forward = edges[0] < edges[n]

        if self.forward:
            self.bin = 0
            self.lo = edges[0]
            self.hi = edges[1]
        else:
            self.bin = n - 1
            self.lo = edges[n]
            self.hi = edges[n - 1]

        self.atend = n < 1

    def increment(self):
        if self.atend:
            raise IndexError("moving beyond final bin")
        self.lo = self.hi
        if self.forward:
            self.bin += 1
            self.atend = self.bin >= self.n
            if not self.atend:
                self.hi = self.edges[self.bin + 1]
        else:
            self.atend = self.bin == 0
            if not self.atend:
                self.bin -= 1
                self.hi = self.edges[self.bin]
        return self


def rebin_counts_portion(Nold, vold, Iold, Nnew, vnew, Inew, ND_portion):
    # Note: inspired by rebin from OpenGenie, but using counts per bin
    # rather than rates.

    # Does not work in place
    if Iold is Inew:
        raise ValueError("does not work in place")

    # Traverse both sets of bin edges; if there is an overlap, add the portion
    # of the overlapping old bin to the new bin.
    _from = BinIter(Nold, vold)
    _to = BinIter(Nnew, vnew)
    while not _from.atend and not _to.atend:
        # std::cout << "from " << from.bin << ": [" << from.lo << ", " << from.hi << "]\n";
        # std::cout << "to " << to.bin << ": [" << to.lo << ", " << to.hi << "]\n";
        if _to.hi <= _from.lo:
            _to.increment()  # new must catch up to old
        elif _from.hi <= _to.lo:
            _from.increment()  # old must catch up to new
        else:
            overlap = min(_from.hi, _to.hi) - max(_from.lo, _to.lo)
            portion = overlap / (_from.hi - _from.lo)
            Inew[_to.bin] += Iold[_from.bin] * portion * ND_portion
            if _to.hi > _from.hi:
                _from.increment()
            else:
                _to.increment()


def rebin_counts_old(Nold, xold, Iold, Nnew, xnew, Inew):
    # Note: inspired by rebin from OpenGenie, but using counts per bin
    # rather than rates.

    # Clear the new bins
    for i in range(Nnew):
        Inew[i] = 0

    rebin_counts_portion(Nold, xold, Iold, Nnew, xnew, Inew, 1.0)


def rebin_counts(xold, Iold, xnew, Inew):
    # Note: inspired by rebin from OpenGenie, but using counts per bin
    # rather than rates.
    Nold = len(Iold)
    Nnew = len(Inew)
    # Clear the new bins
    for i in range(Nnew):
        Inew[i] = 0

    rebin_counts_portion(Nold, xold, Iold, Nnew, xnew, Inew, 1.0)


def rebin_intensity(Nold, xold, Iold, dIold, Nnew, xnew, Inew, dInew):
    # Note: inspired by rebin from OpenGenie, but using counts per bin rather than rates.

    # Clear the new bins
    for i in range(Nnew):
        dInew[i] = Inew[i] = 0

    # Traverse both sets of bin edges; if there is an overlap, add the portion
    # of the overlapping old bin to the new bin.
    _from = BinIter(Nold, xold)
    _to = BinIter(Nnew, xnew)
    while not _from.atend and not _to.atend:
        # std::cout << "from " << from.bin << ": [" << from.lo << ", " << from.hi << "]\n";
        # std::cout << "to " << to.bin << ": [" << to.lo << ", " << to.hi << "]\n";
        if _to.hi <= _from.lo:
            _to.increment()  # new must catch up to old
        elif _from.hi <= _to.lo:
            _from.increment()  # old must catch up to new
        else:
            overlap = min(_from.hi, _to.hi) - max(_from.lo, _to.lo)
            portion = overlap / (_from.hi - _from.lo)

            Inew[_to.bin] += (Iold[_from.bin]) * portion
            # add in quadrature
            dInew[_to.bin] += (dIold[_from.bin] * portion) ** 2
            if _to.hi > _from.hi:
                _from.increment()
            else:
                _to.increment()

    # Convert variance to standard deviation.
    for i in range(Nnew):
        dInew[i] = math.sqrt(dInew[i])


def rebin_counts_2D(xold, yold, Iold, xnew, ynew, Inew):
    Nxold = len(xold) - 1
    Nyold = len(yold) - 1
    Nxnew = len(xnew) - 1
    Nynew = len(ynew) - 1

    # Clear the new bins
    for i in range(Nxnew):
        for j in range(Nynew):
            Inew[i][j] = 0

    # Traverse both sets of bin edges; if there is an overlap, add the portion
    # of the overlapping old bin to the new bin.  Scale this by the portion
    # of the overlap in y.
    _from = BinIter(Nxold, xold)
    _to = BinIter(Nxnew, xnew)

    while not _from.atend and not _to.atend:
        if _to.hi <= _from.lo:
            _to.increment()  # new must catch up to old
        elif _from.hi <= _to.lo:
            _from.increment()  # old must catch up to new
        else:
            overlap = min(_from.hi, _to.hi) - max(_from.lo, _to.lo)
            portion = overlap / (_from.hi - _from.lo)
            rebin_counts_portion(Nyold, yold, Iold[_from.bin], Nynew, ynew, Inew[_to.bin], portion)
            if _to.hi > _from.hi:
                _from.increment()
            else:
                _to.increment()
