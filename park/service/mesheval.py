raise NotImplementedError("Incomplete -- do not use")
"""
Meshgrid function calculator

Evaluates a multi-dimensional function over a grid.

Function dimensions are keyword arguments.

Function arguments and return values are usually numbers, but they don't
need to be.
"""
import numpy

from park.client import JobDescription, make_kernel

def mesheval(f, mesh={}, kwargs={}, returntype='d', vectorized=False, 
             preview=None, server=None):
    """
    Evaluate *f* over a mesh.
    
    *f* is a service kernel or a function within a package.

    *mesh* is a list [('x', [v0,v1,v2,...]), ('y', [v0,v1,v2,...]), ... ] 
    where x,y, ... are the names of the parameters that are to be meshed and 
    [v0,v1,v2, ...] are the grid points in that dimension of the mesh.

    *kwargs* are the values for the non-meshed function arguments. 

    *returntype* is the type of the returned value.  By default it is
    assumed to be a floating point double, or 'f8'.

    *vectorized* is true if *f* accepts array inputs (default)
 
    *preview* is a component to generate a thumbnail plot of the mesh on
    the current figure and return the HTML markup for the figure caption.
    By default this will show a 2D image for the first two dimensions, 
    summed across the remaining dimensions.

    The returned matrix dimensions follow the order given in *mesh*.  Only 
    the first dimension is parallelized, so you should choose the longest
    steps as the first.  If some dimensions are slower than others (e.g., 
    because they cache a table of precalculated values for each lambda in 
    the case of a Poisson random number generator), they should be used 
    earlier in the list of dimensions.

    The return type must be a string describing the returned object. Usually
    it will be on of the simple types::
    
        f4, f8, f12 (single, double, long double)
        c8, c12, c24 (complex single, double, long double)
        i1, i2, i4, i8 (byte, short, int, long)
        u1, u2, u4, u8 (unsigned byte, short, int, long)
        a#, U# (character/unicode string of length #)
        object (python object, e.g., for varying length strings)

    Tuples can be returned by listing the types of the tuple separated
    by commas.  For example, to return a,b,c where a is an integer, b
    is 3x2 doubles and c is a string of length 3 you would use::
    
        returntype='i4,(3,2)f8,a3'

    The returned tuples can be retrieved from the array.  For example,
    to retrieve the 3x2 doubles from mesh i,j use::
    
        result[i,j][1]

    Alternatively, a particular field from the tuple can be retrieved
    as an array using index 'f#' on the result, where # is the number
    of the field.  This is equivalent to the above::
    
        result['f1'][i,j]

    The names can be changed using result.dtype.names = [list of names].

    See the numpy manual for details.[1]
    
    [1] http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    """
    service = dict(name="park.services.mesheval.mesheval_service",
                   input=dict(mesh=mesh, preview=make_kernel(preview)))
    kernel = dict(name="park.services.mesheval.mesheval_worker",
                  input=dict(kernel=make_kernel(f),
                             mesh=mesh, kwargs=kwargs,
                             returntype=returntype,
                             vectorized=vectorized))
    job = JobDescription(requires=[],service=service,kernel=kernel)
    return job.submit(server)

# ==================================================================

def mesheval_worker(env, input):
    kernel = input['kernel']
    fn = env.get_kernel(kernel['name'],kernel['input'])
    return MeshEvalWorker(fn)


# Simplest Mesheval service
# No progress reporting, checkpoint/resume or thumbnail views.
def simple_mesheval_service(env, input):
    mesh = input['mesh']
    _,step = mesh[0] 
    res = env.mapper(step)
    return numpy.asarray(res)

def monitored_mesheval_service(env, input):
    mesh,preview = input['mesh'], input['preview']
    service = MonitoredMeshevalService(mesh,preview)
    return service.run(env)
    _,step = mesh[0] 
    res = env.mapper(step)
    return numpy.asarray(res)

mesheval_service = monitored_mesheval_service



# Complete Mesheval service
# 1. Progress reporting requires that we split the map function in pieces,
# and call ready after each piece.
# 2. Thumbnail views require that we traverse the inputs using start/stride
# and that the starts be out of order
# 3. Checkpoint/resume requires us to keep the start indices and the
# partial mesh in the object state so they can be saved.  Prepare sets
# up the initial state and resume sets up the restored state.  This
# makes the main loop in run ugly since it has to keep track of state.
class MonitoredMeshEvalService:
    """
    Mesheval service with progress reporting, checkpoint/resume and
    thumbnail views.
    """
    def __init__(self, name=None):
        self.name = name

    def prepare(self, handler, request):
        self.request = request
        self.mesh = []
        # Build the mesh from coarse grained to fine grained so that
        # big problems can report partial results.  Make sure that
        # at least 2 inputs are available for each CPU in the pool.
        print "n",len(request.steps[0]),"p",handler.poolsize
        if len(request.steps[0]) > 4*handler.poolsize:
            self.start = [0,5,2,8,3,7,4,9,1,6]
        else:
            self.start = [0]

    def run(self, handler):
        print "starting MeshEval"
        input = self.request.steps[0]
        stride = len(self.start)
        for s in self.start[len(self.mesh):]:
            self.mesh.append(env.mapper(input[s::stride]))
            env.progress(self.progress())
            env.checkpoint(self.checkpoint)

        # Construct the return matrix.  Because we are doing the
        # work in stripes, we need to fill the results in stripes.
        A = numpy.empty([len(v) for v in self.request.steps],
                        self.request.returntype)
        for s,m in zip(self.start,self.mesh):
            #print s,stride,numpy.vstack(m).shape,A.shape
            A[s::stride] = numpy.vstack(m)
        return A

    def checkpoint(self):
        return dict(request=self.request,
                    start=self.start,
                    mesh=self.mesh)

    def resume(self, state, handler):
        self.request = state['request']
        self.start = state['start']
        self.mesh = state['mesh']
        self.run(handler)

    def preview(self):
        """
        Draw a thumbnail image of the progress on the current figure
        and return HTML markup to place below the figure.
        """
        # Build partial matrix
        shape0 = numpy.sum([A.shape[0] for A in self.mesh])
        shape = [shape0]+[len(v) for v in self.request.steps[1:]]
        A = numpy.empty(shape, self.request.returntype)
        stride = len(self.mesh) # drop incomplete stripes
        offset = numpy.argsort(self.start[:len(self.mesh)])
        for s,m in zip(offset,self.mesh):
            A[s::stride] = m
        
        # Construct value vector for partial dimensions
        steps0 = []*shape0
        oldsteps0 = self.request.steps[0]
        oldstride = len(self.start)
        for s,olds in zip(offset, self.start[:len(self.mesh)]):
            steps0[s::stride] = oldsteps0[olds::oldstride]

        # Generate preview on the current matplotlib figure
        dims = self.request.dims
        steps = [steps0] + self.request.steps[1:]
        if self.request.preview is not None:
            caption = self.request.preview(dims, steps, A)
        else:
            if self.request.reduce is not None:
                A = self.request.reduce(dims, steps, A)
            meshplot(dims, steps, A)
            caption = "<p>Partial view of meshed matrix</p>"

        return caption
    
    def progress(self):
        return (numpy.sum([len(v) for v in self.mesh]), 
                len(self.request.steps[0]), "rows")

    def cleanup(self):
        pass


# ==========================================================================
# Helper functions
def meshplot(dims, steps, A):
    import pylab
    from matplotlib.colors import LogNorm
    # Collapse fastest dimensions
    while A.ndim > 2:
        A = numpy.sum(A,axis=-1)

    if A.ndim == 2:
        norm = None
        # If more that two orders of magnitude dynam
        if numpy.any(A != 0):
            lo = numpy.min(abs(A[A!=0]))
            hi = numpy.max(abs(A))
            if numpy.any(A<0):
                maxneg = -numpy.min(A[A<0])
            else:
                maxneg = 0
            if hi >= 100*lo and hi >= 100*maxneg:
                norm = LogNorm(vmin=lo, vmax=hi)
            # Remove invalid values
            A = A+0
            A[A<=lo] = lo/2
        pylab.xlabel(dims[1])
        pylab.ylabel(dims[0])
        pylab.pcolormesh(steps[1], steps[0], A, norm=norm)
        pylab.colorbar()
    else:
        pylab.xlabel(dims[0])
        pylab.ylabel('value')
        pylab.plot(steps[0], A)




class MeshEvalWorker(object):
    def __init__(self, fn, steps, vectorized):
        # Note that we have not done any preprocessing.
        self.fn = fn
        self.steps = steps
        self.v

    def evalpop_vector(self, ypop):
        mesh = meshgridn(ypop,*self.request.steps[1:])
        kwargs = dict(self.request.kwargs) # copy
        kwargs.update(zip(self.request.dims,mesh))
        return self.__f(**kwargs)
    
    def evalpop_scalar(self, ypop):
        if self.__do_prepare:
            self.__do_prepare = False
            self.__f = dill.loads(self.request.fpickle)
        kwargs = dict(self.request.kwargs) # copy
        shape = [len(v) for v in self.request.steps]
        result = numpy.empty([len(ypop)] + shape[1:], self.request.returntype)
        result1d = result.ravel() # Flattened view on the array
        strides = list(reversed(numpy.cumproduct(list(reversed(result.shape[1:])))))
        steps = [ypop] + list(self.request.steps[1:])
        dims = self.request.dims
        ndims = len(dims)
        def _loop(idx,offset):
            # Recursive eval loop...not too bad since the number of
            # dimensions will be small.
            print ">"*idx,"starting"
            dim_name = dims[idx]
            dim_steps = steps[idx]
            if idx == ndims - 1:
                for v in dim_steps:
                    kwargs[dim_name] = v
                    result1d[offset] = self.__f(**kwargs)
                    offset += 1
            else:
                for v in dim_steps:
                    kwargs[dim_name] = v
                    _loop(idx+1,offset)
                    offset += strides[idx]
            print "<"*idx,"completed"
        _loop(0,0) # Start at dimension 0 with offset 0
        return result

    def map(self, ypop):
        print "starting vector",len(ypop)
        if self.request.vectorized:
            res = self.evalpop_vector(ypop)
        else:
            res = self.evalpop_scalar(ypop)
        print "completed vector",len(ypop)
        return res
    def __call__(self, v):
        return numpy.asarray(self.map([v]))

def meshgridn(*args):
    """
    Form a list of grids from a list of vectors.
    
    For vectors d1, d2, ..., dn, the returned grid, Dk, will have elements
    Dk[...,i,...] equal to dk[i] where i is the kth index of Dk.
    """
    dims = [len(v) for v in args]
    grids = [_grid(dims,k,v) for k,v in enumerate(args)]
    return grids

def _grid(dims,k,v):
    """
    Build array Dk for index *k* from vector *v*.
    
    For vectors v, the returned array, Dk, will have elements
    Dk[...,i,...] equal to v[i] where i is the kth index of Dk.
    """
    newidx = [None]*len(dims)
    newidx[k] = slice(None)
    dims = list(dims)  # Make a mutable copy
    dims[k] = 1
    return numpy.tile(numpy.asarray(v)[newidx],dims)
