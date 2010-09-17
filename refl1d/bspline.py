# This program is public domain
"""
BSpline calculator.

Given a set of knots, compute the degree 3 B-spline and any derivatives
that are required.
"""
from __future__ import division
import numpy

def max(a,b):
    return (a<b).choose(a,b)

def min(a,b):
    return (a>b).choose(a,b)

def pbs(x, y, xt, flat=True, parametric=False):
    x = list(sorted(x))
    assert x[0] == 0 and x[-1] == 1
    knot = numpy.hstack((0, 0, numpy.linspace(0,1,len(y)), 1, 1))
    if flat:
        cx = numpy.hstack((x[0],x[0],x[0],(2*x[0]+x[1])/3, 
                           x[1:-1],
                           (2*x[-1]+x[-2])/3, x[-1]))
    else:
        cx = numpy.hstack(([x[0]]*3, x, x[-1]))
    cy = numpy.hstack(([y[0]]*3, y, y[-1]))

    if parametric:
        return _bspline3(knot,cx,xt),_bspline3(knot,cy,xt)

    # Find parametric t values corresponding to given z values
    # First try a few newton steps
    t = numpy.interp(xt,x,numpy.linspace(0,1,len(x)))
    for _ in range(6):
        Pt,dPt = _bspline3(knot,cx,t,nderiv=1)
        idx = dPt!=0
        t[idx] = (t - (Pt-xt)/dPt)[idx]
    # Use bisection when newton failed
    idx = numpy.isnan(t) | (abs(_bspline3(knot,cx,t)-xt)>1e-9)
    if idx.any():
        missing = xt[idx]
        #print missing
        t_lo, t_hi = 0*missing, 1*missing
        for _ in range(30): # bisection with about 1e-9 tolerance
            trial = (t_lo+t_hi)/2
            Ptrial = _bspline3(knot,cx,trial)
            tidx = Ptrial<missing
            t_lo[tidx] = trial[tidx]
            t_hi[~tidx] = trial[~tidx]
        t[idx] = (t_lo+t_hi)/2
    #print "err",numpy.max(abs(_bspline3(knot,cx,t)-xt))

    # Return y evaluated at the interpolation points
    return _bspline3(knot,cx,t), _bspline3(knot,cy,t)

def bspline(y, xt, flat=True):
    """
    Evaluate the B-spline specified by the given knot sequence and
    control values at the parametric points t.

    If clamp is True, the spline value is clamped to the value of
    the final control point beyond the ends of the knot sequence.
    If clamp is False, the spline will go to zero at +/- infinity.
    """
    if flat:
        knot = numpy.hstack((0, 0, numpy.linspace(0,1,len(y)), 1, 1))
        cy = numpy.hstack(([y[0]]*3, y, y[-1]))
    else: 
        raise NotImplementedError
        # The following matches first derivative but not second
        knot = numpy.hstack((0, 0, numpy.linspace(0,1,len(y)), 1, 1))
        cy = numpy.hstack((y[0], y[0], y[0],
                           y[0] + (y[1]-y[0])/3,
                           y[1:-1],
                           y[-1] + (y[-2]-y[-1])/3, y[-1]))
    return _bspline3(knot,cy,xt)
    
def _bspline3(knot,control,t,nderiv=0):
    knot,control,t = [numpy.asarray(v) for v in knot, control, t]

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



def speed_test():
    import time
    x = linspace(0,1,7)
    x[1],x[-2] = x[2],x[-3]
    y = [9,11,2,3,8,0,2]
    t = linspace(0,1,400)
    t0 = time.time()
    for i in range(1000): bspline(y,t,flat=True)
    print "bspline (ms)",(time.time()-t0)/1000

def demo():
    from pylab import hold, linspace, plot, show
    hold(True)
    x = linspace(0,1,7)
    #x[1],x[-2] = x[0],x[-1] 
    x[1],x[-2] = x[2],x[-3]
    x[1],x[-2] = x[2]+0.01,x[-3]-0.01
    #x[1],x[-2] = x[1]-x[1]/2,x[-1]-x[1]/2
    #y = [9,6,1,3,8,4,2]
    #y = [9,11,13,3,-2,0,2]
    y = [9,11,2,3,8,0,2]
    #y = [9,9,1,3,8,2,2]
    t = linspace(0,1,400)
    plot(linspace(x[0],x[-1],len(x)),y,':oy')
    #plot(xt,bspline(y,t,flat=False),'-.y') # bspline
    plot(t,bspline(y,t,flat=True),'-y') # bspline

    yt = pbs(x,y,t,flat=False)
    plot(t,yt,'-.b') # pbs
    yt = pbs(x,y,t,flat=True)
    plot(t,yt,'-b') # pbs
    plot(sorted(x),y,':ob')
    show()
    
if __name__ == "__main__":
    #demo()
    speed_test()