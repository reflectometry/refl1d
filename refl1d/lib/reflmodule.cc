/* This program is public domain. */

#include <Python.h>
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


#define MODULE_DOC "Reflectometry C Library"
#define MODULE_NAME "reflmodule"
#define MODULE_INIT2 initreflmodule
#define MODULE_INIT3 PyInit_reflmodule
#define MODULE_METHODS methods

/* ==== boilerplate python 2/3 interface bootstrap ==== */

#if defined(WIN32) && !defined(__MINGW32__)
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

#if PY_MAJOR_VERSION >= 3

  DLL_EXPORT PyMODINIT_FUNC MODULE_INIT3(void)
  {
    static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      MODULE_NAME,         /* m_name */
      MODULE_DOC,          /* m_doc */
      -1,                  /* m_size */
      MODULE_METHODS,      /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
    };
    return PyModule_Create(&moduledef);
  }

#else /* PY_MAJOR_VERSION >= 3 */

  DLL_EXPORT PyMODINIT_FUNC MODULE_INIT2(void)
  {
    Py_InitModule4(MODULE_NAME,
		 MODULE_METHODS,
		 MODULE_DOC,
		 0,
		 PYTHON_API_VERSION
		 );
  }

#endif /* PY_MAJOR_VERSION >= 3 */
