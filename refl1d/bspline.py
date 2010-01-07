# This program is public domain
"""
BSpline calculator.

Given a set of knots, compute the degree 3 Bspline and any derivatives
that are required.
"""
import numpy

def max(a,b):
    return (a<b).choose(a,b)

def min(a,b):
    return (a>b).choose(a,b)

def bspline3(knot,
             control,
             t,
             clamp=True,
             nderiv=0
             ):
    """
    Evaluate the B-spline specified by the given knot sequence and
    control values at the parametric points t.   The knot sequence
    should be four elements longer than the control sequence.

    If nderiv is greater than 0, returns derivatives as well as the
    function value.  For example, nderiv=3 returns f,f',f'',f''',

    If clamp is True, the spline value is clamped to the value of
    the final control point beyond the ends of the knot sequence.
    If clamp is False, the spline will go to zero at +/- infinity
    as in the traditional algorithm.
    """
    degree = len(knot) - len(control);
    if degree != 4:
        raise ValueError, "must have two extra knots at each end"

    if clamp:
        # Alternative approach spline is clamped to initial/final control values
        control = numpy.concatenate(([control[0]]*(degree-1),
                                     control, [control[-1]]))
    else:
        # Traditional approach: spline goes to zero at +/- infinity.
        control = numpy.concatenate(([0]*(degree-1), control, [0]))


    # Deal with values outside the range
    valid = (t > knot[0]) & (t <= knot[-1])
    tv  = t[valid]
    f   = numpy.zeros(t.shape)
    f[t<=knot[0]]  = control[0]
    f[t>=knot[-1]] = control[-1]

    # Find B-Spline parameters for the individual segments
    end     = len(knot)-1
    segment = knot.searchsorted(tv)-1
    tm2 = knot[max(segment-2,0)]
    tm1 = knot[max(segment-1,0)]
    tm0 = knot[max(segment-0,0)]
    tp1 = knot[min(segment+1,end)]
    tp2 = knot[min(segment+2,end)]
    tp3 = knot[min(segment+3,end)]

    P4 = control[min(segment+3,end)]
    P3 = control[min(segment+2,end)]
    P2 = control[min(segment+1,end)]
    P1 = control[min(segment+0,end)]

    # Compute second and third derivatives.
    if nderiv > 1:
        # Normally we require a recursion for Q, R and S to compute
        # df, d2f and d3f respectively, however Q can be computed directly
        # from intermediate values of P, S has a recursion of depth 0,
        # which leaves only the R recursion of depth 1 in the calculation
        # below.
        Q4 = (P4 - P3) * 3 / (tp3-tm0)
        Q3 = (P3 - P2) * 3 / (tp2-tm1)
        Q2 = (P2 - P1) * 3 / (tp1-tm2)
        R4 = (Q4 - Q3) * 2 / (tp2-tm0)
        R3 = (Q3 - Q2) * 2 / (tp1-tm1)
        if nderiv > 2:
            S4 = (R4 - R3) / (tp1-tm0)
            d3f = numpy.zeros(t.shape)
            d3f[valid] = S4
        R4 = ( (tv-tm0)*R4 + (tp1-tv)*R3 ) / (tp1 - tm0)
        d2f = numpy.zeros(t.shape)
        d2f[valid] = R4

    # Compute function value and first derivative
    P4 = ( (tv-tm0)*P4 + (tp3-tv)*P3 ) / (tp3 - tm0)
    P3 = ( (tv-tm1)*P3 + (tp2-tv)*P2 ) / (tp2 - tm1)
    P2 = ( (tv-tm2)*P2 + (tp1-tv)*P1 ) / (tp1 - tm2)
    P4 = ( (tv-tm0)*P4 + (tp2-tv)*P3 ) / (tp2 - tm0)
    P3 = ( (tv-tm1)*P3 + (tp1-tv)*P2 ) / (tp1 - tm1)
    if  nderiv >= 1:
        df = numpy.zeros(t.shape)
        df[valid] = (P4-P3) * 3 / (tp1-tm0)
    P4 = ( (tv-tm0)*P4 + (tp1-tv)*P3 ) / (tp1 - tm0)
    f[valid]  = P4

    if   nderiv == 0: return f
    elif nderiv == 1: return f,df
    elif nderiv == 2: return f,df,d2f
    else:             return f,df,d2f,d3f
