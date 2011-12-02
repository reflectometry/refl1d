/* This program is public domain. */

//#include <Python.h>
#include "methods.h"

//#define PY_ARRAY_UNIQUE_SYMBOL reflmodule_array
//#include <numpy/arrayobject.h>

static PyMethodDef methods[] = {

	{"_reflectivity_amplitude",
	 Preflectivity_amplitude,
	 METH_VARARGS,
	 "_reflectivity_amplitude(d,sigma,rho,irho,Q,rho_offset,R): compute reflectivity putting it into vector R of len(Q)"},

	{"_magnetic_amplitude",
	 Pmagnetic_amplitude,
	 METH_VARARGS,
	 "_magnetic_amplitude(d,sigma,rho,irho,rhoM,expth,Q,rho_offset,R1,R2,R3,R4): compute amplitude putting it into vector R of len(Q)"},

	{"_contract_by_area",
         Pcontract_by_area,
         METH_VARARGS,
         "_contract_by_area(d,sigma,rho,irho,dA): join layers in microstep profile, keeping error under control"},

         {"_contract_mag",
          Pcontract_mag,
          METH_VARARGS,
          "_contract_mag(d,sigma,rho,irho,rhoM,thetaM,dA): join layers in microstep profile, keeping error under control"},

        {"_contract_by_step",
         Pcontract_by_step,
         METH_VARARGS,
         "_contract_by_step(d,sigma,rho,irho,dv): join layers in microstep profile, keeping error under control"},

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
  //PY_ARRAY_UNIQUE_SYMBOL = NULL;
  //import_array();
}
