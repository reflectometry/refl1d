from __future__ import division
__all__ = ['plot_all', 'plot_corr', 'plot_corrmatrix',
           'plot_trace', 'plot_vars', 'plot_var',
           'plot_R','plot_logp']
import math
import numpy
from numpy import arange, squeeze, reshape, linspace, meshgrid, vstack, NaN, inf
import pylab
from pylab import (axes, plot, MaxNLocator, setp, title, xlabel, 
                   ylabel, legend, figure, show)
from . import corrplot
from .stats import credible_interval, stats
from .formatnum import format_uncertainty
from .util import console

def plot_all(state, portion=None):
    
    figure(1); plot_vars(state, portion=portion)
    figure(2); plot_trace(state, portion=portion)
    figure(3); plot_R(state, portion=portion)
    figure(4); plot_logp(state, portion=portion)
    figure(5); plot_corrmatrix(state, portion=portion)
    show()

def plot_var(state, var=0, portion=None, selection=None, **kw):
    points, logp = state.sample(portion=portion, vars=[var], 
                                selection=selection)
    _plot_var(points.flatten(), logp, label=state.labels[var], **kw)

def plot_vars(state, vars=None, portion=None, selection=None, **kw):
    points, logp = state.sample(portion=portion, vars=vars, selection=selection)
    if vars==None:
        vars = range(points.shape[1])
    nw,nh = tile_axes(len(vars))
    for k,v in enumerate(vars):
        pylab.subplot(nw,nh,k+1)
        _plot_var(points[:,k].flatten(), logp, label=state.labels[v], **kw)

def tile_axes(n, size=None):
    """
    Creates a tile for the axes which covers as much area of the graph as
    possible while keeping the plot shape near the golden ratio.
    """
    if size == None:
        size = pylab.gcf().get_size_inches()
    figwidth, figheight = size
    # Golden ratio phi is the preferred dimension
    #    phi = sqrt(5)/2
    #
    # nw, nh is the number of tiles across and down respectively
    # w, h are the sizes of the tiles
    #
    # w,h = figwidth/nw, figheight/nh
    #
    # To achieve the golden ratio, set w/h to phi:   
    #     w/h = phi  => figwidth/figheight*nh/nw = phi
    #                => nh/nw = phi * figheight/figwidth
    # Must have enough tiles:
    #     nh*nw > n  => nw > n/nh
    #                => nh**2 > n * phi * figheight/figwidth
    #                => nh = floor(sqrt(n*phi*figheight/figwidth))
    #                => nw = ceil(n/nh)
    phi = math.sqrt(5)/2
    nh = int(math.floor(math.sqrt(n*phi*figheight/figwidth)))
    if nh<1: nh = 1
    nw = int(math.ceil(n/nh))
    return nw,nh

    
def _plot_var(points, logp, label="P", nbins=50, ci=0.95):
    # Sort the data
    idx = numpy.argsort(points)
    points = points[idx]
    logp=logp[idx]
    idx = numpy.argmax(logp)
    maxlogp = logp[idx]
    best = points[idx]

    # If weighted, use the relative probability from the marginal distribution
    # as the weight
    #weights = numpy.exp(logp-maxlogp) if weighted else None
    weights = None
    
    # Choose the interval for the histogram
    ONE_SIGMA = 0.15865525393145705
    rangeci,range68 = credible_interval(x=points, weights=weights, 
                                        ci=[ci,1-2*ONE_SIGMA])

    # Compute stats
    median = points[int(len(points)/2)]
    mean, std = stats(x=points, weights=weights)
    
    # Produce a histogram
    hist, bins = numpy.histogram(points, bins=nbins, range=rangeci,
                                 #new=True,
                                 normed=True, weights=weights)

    # Find the max likelihood for values in this bin
    edge = numpy.searchsorted(points,bins) 
    histbest = [numpy.max(logp[edge[i]:edge[i+1]]) 
                if edge[i]<edge[i+1] else -inf
                for i in range(nbins)]
    density = kde_1d(points)

    # scale to marginalized probability with peak the same height as hist
    histbest = numpy.exp(histbest-maxlogp)
    histbest *= numpy.max(hist)

    # Make sure numbers are formatted with the appropriate precision
    scale = 10**int(numpy.log10(rangeci[1]-rangeci[0])-2)
    def format(x): return "%g"%(numpy.round(x/scale)*scale)
    statsbox = """\
mean   = %s
median = %s
best   = %s
68%% interval  = [%s %s]
%g%% interval  = [%s %s]\
"""%(format_uncertainty(mean,std), format(median), format(best),
     format(range68[0]), format(range68[1]),
     100*ci, format(rangeci[0]), format(rangeci[1]))

    # Plot the histogram
    pylab.bar(bins[:-1], hist, width=bins[1]-bins[0])

    # Plot the kernel density estimate
    #x = linspace(bins[0],bins[-1],100)
    #pylab.plot(x, density(x), '-k', hold=True)
    
    # Plot the marginal maximum likelihood
    centers = (bins[:-1]+bins[1:])/2
    pylab.plot(centers, histbest, '-g', hold=True)
    # Shade things inside 1-sigma
    pylab.axvspan(range68[0],range68[1],alpha=0.1)
    pylab.axvline(median)
    pylab.axvline(mean)
    pylab.axvline(best)
    if 0:
        pylab.text(0.01, 0.95, statsbox, 
                   backgroundcolor=(1,1,0,0.2),
                   verticalalignment='top',
                   horizontalalignment='left',
                   transform=pylab.gca().transAxes)
    else:
        pylab.text(0.01, 0.95, label, 
                   backgroundcolor=(1,1,0,0.2),
                   verticalalignment='top',
                   horizontalalignment='left',
                   transform=pylab.gca().transAxes)
        print "Parameter",label
        print statsbox
    pylab.xlabel(label)


def plot_corrmatrix(state, vars=None, portion=None, selection=None):
    points, _ = state.sample(portion=portion, vars=vars, selection=selection)
    labels = state.labels if vars==None else [state.labels[v] for v in vars]
    c = corrplot.Corr2d(points.T, bins=50, labels=labels)
    c.plot()
    print "Correlation matrix\n",c.R()


from scipy.stats import kde
class kde_1d(kde.gaussian_kde):
    covariance_factor = lambda self: 2*self.silverman_factor()

class kde_2d(kde.gaussian_kde):
    covariance_factor = kde.gaussian_kde.silverman_factor
    def __init__(self, dataset):
        kde.gaussian_kde.__init__(self, dataset.T)
    def evalxy(self, x, y):
        X,Y = meshgrid(x,y)
        dxy = self.evaluate(vstack([X.flatten(),Y.flatten()]))
        return dxy.reshape(X.shape)
    __call__ = evalxy
    
def plot_corr(state, vars=(0,1), portion=None, selection=None):
    p1,p2 = vars
    labels = [state.labels[v] for v in vars]
    points, _ = state.sample(portion=portion, vars=vars, selection=selection)
    p1,p2 = 0,1
    
    # Form kernel density estimates of the parameters
    xmin,xmax = min(points[:,p1]),max(points[:,p1])
    density_x = kde_1d(points[:,p1])
    x = linspace(xmin, xmax, 100)
    px = density_x(x)
    
    density_y = kde_1d(points[:,p2])
    ymin,ymax = min(points[:,p2]),max(points[:,p2])
    y = linspace(ymin, ymax, 100)
    py = density_y(y)

    nbins = 50
    axData = axes([0.1,0.1,0.63,0.63]) # x,y,w,h

    #density_xy = kde_2d(points[:,vars])
    #dxy = density_xy(x,y)*points.shape[0]
    #axData.pcolorfast(x,y,dxy,cmap=cm.gist_earth_r) #@UndefinedVariable
    
    axData.plot(points[:,p1], points[:,p2], 'k.', markersize=1)
    axData.set_xlabel(labels[p1])
    axData.set_ylabel(labels[p2])
    axHistX = axes([0.1,0.75,0.63,0.2],sharex=axData)
    axHistX.hist(points[:,p1],nbins,orientation='vertical',normed=1)
    axHistX.plot(x,px,'k-')
    axHistX.yaxis.set_major_locator(MaxNLocator(4,prune="both"))
    setp(axHistX.get_xticklabels(), visible=False,)
    axHistY = axes([0.75,0.1,0.2,0.63],sharey=axData)
    axHistY.hist(points[:,p2],nbins,orientation='horizontal',normed=1)
    axHistY.plot(py,y,'k-')
    axHistY.xaxis.set_major_locator(MaxNLocator(4,prune="both"))
    setp(axHistY.get_yticklabels(), visible=False)

def plot_trace(state, var=0, portion=None):
    if portion == None:
        portion = 0.8 if state.cycle < 1 else 1
    draw, points, _ = state.chains()
    start = int((1-portion)*len(draw))
    plot(arange(start,len(points))*state.thinning,
         squeeze(points[start:,:,var]))
    title('Parameter history for variable %d'%(var+1))
    xlabel('Generation number')
    ylabel('Parameter value')

def plot_R(state, portion=None):
    if portion == None:
        portion = 0.8 if state.cycle < 1 else 1
    draw, R = state.R_stat()
    start = int((1-portion)*len(draw))
    plot(arange(start,len(R)), R[start:])
    title('Convergence history')
    legend(['P%d'%i for i in range(1,R.shape[1]+1)])
    xlabel('Generation number')
    ylabel('R')
    
def plot_logp(state, portion=None):
    if portion == None:
        portion = 0.8 if state.cycle < 1 else 1
    draw, logp = state.logp()
    start = int((1-portion)*len(draw))
    plot(arange(start,len(logp)), logp[start:], '.', markersize=1)
    title(r'Log Likelihood History')
    xlabel('Generation number')
    ylabel('Log likelihood at x[k]')


def _this_is_cruft__points(state, portion, vars=None, selection=None):
    draw, points, logp = state.chains()
    labels = state.labels
    start = int((1-portion)*len(draw))
    points = points[start:]
    logp = logp[start:]    
    Ngen,Npop,Nvar = points.shape
    points = reshape(points,(Ngen*Npop,Nvar))
    logp = reshape(logp,(Ngen*Npop))
    if labels == None:
        labels = ["P"+str(i+1) for i in range(points.shape[-1])]
    if selection is not None:
        idx = True
        for v,r in selection.items():
            idx = idx & (points[:,v]>=r[0]) & (points[:,v]<=r[1])
        points = points[idx,:]
        logp = logp[idx]
    if vars != None:
        points = points[:,vars]
        labels = [labels[i] for i in vars]
    return points, logp, labels

