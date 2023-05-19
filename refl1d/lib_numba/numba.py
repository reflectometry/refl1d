"""
A shim library in case numba is not installed

(for some environments, numba cannot be installed, e.g. pyodide)
"""
try:
    from numba import (
        njit,
        int8,
        int32,
        int64,
        float32,
        float64,
        complex128,
        boolean,
        prange,
        experimental
    )
except ImportError:
    import warnings

    warnings.warn("Numba is not installed - calculations will be slow")

    int8 = []
    int32 = []
    int64 = []
    float32 = []
    float64 = []
    complex128 = []
    boolean = []

    prange = range

    def passthrough_decorator(*args, **kw):
        return lambda f: f

    njit = passthrough_decorator


    class experimental:
        jitclass = passthrough_decorator