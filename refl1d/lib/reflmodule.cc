/* This program is public domain. */

#include "methods.h"

#define PY_ARRAY_UNIQUE_SYMBOL reflmodule_array
#include <numpy/arrayobject.h>

PyObject* pyvector(int n, double v[])
{
   int dims[1];
   dims[0] = n;
   return PyArray_FromDimsAndData(1,dims,PyArray_DOUBLE,(char *)v);
}

static PyMethodDef methods[] = {
	{"_fixedres",
	 Pfixedres,
	 METH_VARARGS,
	 "_fixedres(L,dLoL,dT,Q,dQ): compute resolution for fixed slits at Q given L, dL/L and dT"},

	{"_varyingres",
	 Pvaryingres,
	 METH_VARARGS,
	 "_varyingres(L,dLoL,dToT,Q,dQ): compute resolution for varying slits at Q given L, dL/L and dT/T"},

	{"_convolve",
	 Pconvolve,
	 METH_VARARGS,
	 "_convolve(Qi,Ri,Q,dQ,R): compute convolution of width dQ[k] at points Q[k], returned in R[k]"},

	{"_reflectivity_amplitude",
	 Preflamp,
	 METH_VARARGS, 
	 "_reflectivity_amplitude(rho,mu,d,L,Q,R): compute reflectivity putting it into vector R of len(Q)"},

	{"_magnetic_amplitude",
	 Pmagnetic_amplitude,
	 METH_VARARGS, 
	 "_magnetic_amplitude(rho,mu,d,L,P,expth,Q,R1,R2,R3,R4): compute amplitude putting it into vector R of len(Q)"}, 

	{"_reflectivity_amplitude_rough",
	 Preflamp_rough,
	 METH_VARARGS, 
	 "_refl(rho,mu,d,sigma,L,Q,R): compute reflectivity with approximate roughness putting it into vector R of len(Q)"},

	{"_erf",
	 Perf,
	 METH_VARARGS,
	 "erf(data, result): get the erf of a set of data points"},

	{0}
} ;


#if defined(WIN32) && !defined(__MINGW32__)
__declspec(dllexport)
#endif

	
extern "C" void initreflmodule(void) {
  Py_InitModule4("reflmodule",
		 methods,
		 "Reflectometry C Library",
		 0,
		 PYTHON_API_VERSION
		 );
  PY_ARRAY_UNIQUE_SYMBOL = NULL;
  import_array();
}

