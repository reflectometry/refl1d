/* This program is public domain. */

/* Visual Studio 2008 is missing cstdint */
#if defined(_MSC_VER) && _MSC_VER<=1600
  typedef signed char        int8_t;
  typedef short              int16_t;
  typedef int                int32_t;
  typedef long long          int64_t;
  typedef unsigned char      uint8_t;
  typedef unsigned short     uint16_t;
  typedef unsigned int       uint32_t;
  typedef unsigned long long uint64_t;
#endif

//#define Py_LIMITED_API 0x03020000
#include <Python.h>

#include "methods.h"
#include "rebin.h"
#include "rebin2D.h"


template <typename T>
PyObject* Prebin(PyObject *obj, PyObject *args)
{
  PyObject *in_obj,*Iin_obj,*out_obj,*Iout_obj;
  Py_ssize_t nin,nIin, nout, nIout;
  double *in, *out;
  T *Iin, *Iout;
  DECLARE_VECTORS(4);

  if (!PyArg_ParseTuple(args, "OOOO:rebin",
                        &in_obj,&Iin_obj,&out_obj,&Iout_obj)) return NULL;
  INVECTOR(in_obj,in,nin);
  INVECTOR(Iin_obj,Iin,nIin);
  INVECTOR(out_obj,out,nout);
  OUTVECTOR(Iout_obj,Iout,nIout);
  if (nin-1 != nIin || nout-1 != nIout) {
    PyErr_SetString(PyExc_ValueError,
        "_reduction.rebin: must have one more bin edges than bins");
    FREE_VECTORS();
    return NULL;
  }
  rebin_counts<T>(nin-1,in,Iin,nout-1,out,Iout);
  FREE_VECTORS();
  return Py_BuildValue("");
}

template <typename T>
PyObject* Prebin2d(PyObject *obj, PyObject *args)
{
  PyObject *xin_obj, *yin_obj, *Iin_obj;
  PyObject *xout_obj, *yout_obj, *Iout_obj;
  Py_ssize_t nxin, nyin, nIin;
  Py_ssize_t nxout, nyout, nIout;
  double *xin,*yin,*xout,*yout;
  T *Iin, *Iout;
  DECLARE_VECTORS(6);

  if (!PyArg_ParseTuple(args, "OOOOOO:rebin",
                        &xin_obj, &yin_obj, &Iin_obj,
                        &xout_obj, &yout_obj, &Iout_obj))
        return NULL;

  INVECTOR(xin_obj,xin,nxin);
  INVECTOR(yin_obj,yin,nyin);
  INVECTOR(Iin_obj,Iin,nIin);
  INVECTOR(xout_obj,xout,nxout);
  INVECTOR(yout_obj,yout,nyout);
  OUTVECTOR(Iout_obj,Iout,nIout);
  if ((nxin-1)*(nyin-1) != nIin || (nxout-1)*(nyout-1) != nIout) {
    /* printf("%ld %ld %ld %ld %ld %ld\n",nxin,nyin,nIin,nxout,nyout,nIout); */
    PyErr_SetString(PyExc_ValueError,
        "_reduction.rebin2d: must have one more bin edges than bins");
    FREE_VECTORS();
    return NULL;
  }
  rebin_counts_2D<T>(nxin-1,xin,nyin-1,yin,Iin,
      nxout-1,xout,nyout-1,yout,Iout);
  FREE_VECTORS();
  return Py_BuildValue("");
}

static PyMethodDef methods[] = {

	{"_reflectivity_amplitude",
	 Preflectivity_amplitude,
	 METH_VARARGS,
	 "_reflectivity_amplitude(d,sigma,rho,irho,Q,rho_offset,R): compute reflectivity putting it into vector R of len(Q)"},

	{"_magnetic_amplitude",
	 Pmagnetic_amplitude,
	 METH_VARARGS,
	 "_magnetic_amplitude(d,sigma,rho,irho,sld_b,U1,U3,Aguide,Q,rho_offset,R1,R2,R3,R4): compute amplitude putting it into vector R of len(Q)"},

	{"_calculate_u1_u3",
	 Pcalculate_u1_u3,
	 METH_VARARGS,
	 "_calculate_u1_u3(H,rhoM,thetaM,Aguide,sld_b,U1,U3): compute sld_b,U1,U3 from rhoM,thetaM to pass to magnetic amplitude"},

	{"_contract_by_area",
	 Pcontract_by_area,
	 METH_VARARGS,
	 "_contract_by_area(d,sigma,rho,irho,dA): join layers in microstep profile, keeping error under control"},

	{"_contract_mag",
	 Pcontract_mag,
	 METH_VARARGS,
	 "_contract_mag(d,sigma,rho,irho,rhom,thetam,da): join layers in microstep profile, keeping error under control"},

	{"_contract_by_step",
	 Pcontract_by_step,
	 METH_VARARGS,
	 "_contract_by_step(d,sigma,rho,irho,dv): join layers in microstep profile, keeping error under control"},

	{"_align_magnetic",
	 Palign_magnetic,
	 METH_VARARGS,
	 "_align_magnetic(d,sigma,rho,irho,dm,sigmam,rhom,thetam,result): align the interfaces between nuclear and magnetic profiles"},

	{"convolve",
	 Pconvolve,
	 METH_VARARGS,
	 "convolve(xi,yi,x,dx,y): compute convolution of width dx[k] at points x[k],\nreturned in y[k]"},

	{"convolve_sampled",
	 Pconvolve_sampled,
	 METH_VARARGS,
	 "convolve_sampled(xi,yi,xp,yp,x,dx,y): compute convolution with sampled\ndistribution of width dx[k] at points x[k], returned in y[k]"},

	{"rebin_uint8",
	 &Prebin<uint8_t>,
	 METH_VARARGS,
	 "rebin_uint8(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint8",
	 &Prebin2d<uint8_t>,
	 METH_VARARGS,
	 "rebin2d_uint8(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_uint16",
	 &Prebin<uint16_t>,
	 METH_VARARGS,
	 "rebin_uint16(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint16",
	 &Prebin2d<uint16_t>,
	 METH_VARARGS,
	 "rebin2d_uint16(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_uint32",
	 &Prebin<uint32_t>,
	 METH_VARARGS,
	 "rebin_uint32(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint32",
	 &Prebin2d<uint32_t>,
	 METH_VARARGS,
	 "rebin2d_uint32(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_uint64",
	 &Prebin<uint64_t>,
	 METH_VARARGS,
	 "rebin_uint64(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint64",
	 &Prebin2d<uint64_t>,
	 METH_VARARGS,
	 "rebin2d_uint64(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_float32",
	 &Prebin<float>,
	 METH_VARARGS,
	 "rebin_float32(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_float32",
	 &Prebin2d<float>,
	 METH_VARARGS,
	 "rebin2d_float32(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_float64",
	 &Prebin<double>,
	 METH_VARARGS,
	 "rebin_float64(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_float64",
	 &Prebin2d<double>,
	 METH_VARARGS,
	 "rebin2d_float64(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},

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
