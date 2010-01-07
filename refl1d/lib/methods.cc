/* This program is public domain. */

#include "src/reflcalc.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "methods.h"


#if defined(PY_VERSION_HEX) &&  (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif


#undef BROKEN_EXCEPTIONS


PyObject* Pconvolve(PyObject *obj, PyObject *args)
{
  PyObject *Qi_obj,*Ri_obj,*Q_obj,*dQ_obj,*R_obj;
  const double *Qi, *Ri, *Q, *dQ;
  double *R;
  Py_ssize_t nQi, nRi, nQ, ndQ, nR;
  
  if (!PyArg_ParseTuple(args, "OOOOO:resolution", 
			&Qi_obj,&Ri_obj,&Q_obj,&dQ_obj,&R_obj)) return NULL;
  INVECTOR(Qi_obj,Qi,nQi);
  INVECTOR(Ri_obj,Ri,nRi);
  INVECTOR(Q_obj,Q,nQ);
  INVECTOR(dQ_obj,dQ,ndQ);
  OUTVECTOR(R_obj,R,nR);
  if (nQi != nRi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "_librefl.convolve: Qi and Ri have different lengths");
#endif
    return NULL;
  }
  if (nQ != ndQ || nQ != nR) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "_librefl.convolve: Q, dQ and R have different lengths");
#endif
    return NULL;
  }
  resolution(nQi,Qi,Ri,nQ,Q,dQ,R);
  return Py_BuildValue("");
}


PyObject* Pfixedres(PyObject*obj,PyObject*args)
{
  PyObject *Q_obj,*dQ_obj;
  const double *Q;
  double *dQ;
  double L, dLoL, dT;
  Py_ssize_t nQ, ndQ;

  if (!PyArg_ParseTuple(args, "dddOO:fixedres", 
			&L,&dLoL,&dT,&Q_obj,&dQ_obj)) return NULL;
  INVECTOR(Q_obj,Q,nQ);
  OUTVECTOR(dQ_obj,dQ,ndQ);
  if (nQ != ndQ) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "_librefl.fixedres: Q and dQ have different lengths");
#endif
    return NULL;
  }
  resolution_fixed(L,dLoL,dT,nQ,Q,dQ);
  return Py_BuildValue("");
}


PyObject* Pvaryingres(PyObject*obj,PyObject*args)
{
  PyObject *Q_obj,*dQ_obj;
  const double *Q;
  double *dQ;
  double L, dLoL, dToT;
  Py_ssize_t nQ, ndQ;

  if (!PyArg_ParseTuple(args, "dddOO:fixedres", 
			&L,&dLoL,&dToT,&Q_obj,&dQ_obj)) return NULL;
  INVECTOR(Q_obj,Q,nQ);
  OUTVECTOR(dQ_obj,dQ,ndQ);
  if (nQ != ndQ) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "_librefl.fixedres: Q and dQ have different lengths");
#endif
    return NULL;
  }
  resolution_varying(L,dLoL,dToT,nQ,Q,dQ);
  return Py_BuildValue("");
}


PyObject* Preflamp(PyObject*obj,PyObject*args)
{
  PyObject *Q_obj,*R_obj,*d_obj,*rho_obj,*mu_obj, *wavelength_obj;
  const double *Q, *d, *rho, *mu, *wavelength;
  refl_complex *R;
  Py_ssize_t nQ, nR, nd, nrho, nmu, nwavelength;

  if (!PyArg_ParseTuple(args, "OOOOOO:reflamp", 
			&rho_obj,&mu_obj,&d_obj,&wavelength_obj,&Q_obj,&R_obj)) return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(mu_obj,mu,nmu);
  INVECTOR(Q_obj,Q,nQ);
  INVECTOR(wavelength_obj, wavelength, nwavelength);
  OUTVECTOR(R_obj,R,nR);
  if (nd != nrho || nd != nmu) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,rho,mu have different lengths");
#endif
    return NULL;
  }
  if (nQ != nR || nwavelength != nQ) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "Q,R,wavelength have different lengths");
#endif
    return NULL;
  }
  reflectivity_amplitude(nd, d, rho, mu, wavelength, nQ, Q, R);
  return Py_BuildValue("");
}


PyObject* Pmagnetic_amplitude(PyObject*obj,PyObject*args)
{
  PyObject *Q_obj,*wavelength_obj,*R1_obj,*R2_obj,*R3_obj,*R4_obj,*d_obj,*rho_obj,*mu_obj, *expth_obj, *P_obj;
  const double *Q, *d, *rho, *mu, *P, *wavelength;
  double Aguide;
  Py_ssize_t nQ, nR1, nR2, nR3, nR4, nd, nrho, nmu, nexpth, nP, nwavelength;
  const refl_complex *expth;
  refl_complex *R1, *R2, *R3, *R4;

  if (!PyArg_ParseTuple(args, "OOOOOOdOOOOO:reflectivity", 
			                  &rho_obj, &mu_obj, &d_obj, &wavelength_obj,
												&P_obj,&expth_obj, &Aguide,&Q_obj,
												&R1_obj,&R2_obj,&R3_obj,&R4_obj)) 
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(mu_obj,mu,nmu);
  INVECTOR(P_obj,P,nP);
  INVECTOR(expth_obj,expth,nexpth);
  INVECTOR(Q_obj,Q,nQ);
	INVECTOR(wavelength_obj, wavelength, nwavelength);
  OUTVECTOR(R1_obj,R1,nR1);
  OUTVECTOR(R2_obj,R2,nR2);
  OUTVECTOR(R3_obj,R3,nR3);
  OUTVECTOR(R4_obj,R4,nR4);
  if (nd != nrho || nd != nmu || nd != nP || nd != nexpth) {
#ifndef BROKEN_EXCEPTIONS
    printf("%ld %ld %ld %ld %ld\n", 
	long(nd), long(nrho), long(nmu), long(nP), long(nexpth));
    PyErr_SetString(PyExc_ValueError, "d,rho,mu,P,expth have different lengths");
#endif
    return NULL;
  }
  if (nQ != nR1 || nQ != nR2 || nQ != nR3 || nQ != nR4 || nwavelength != nQ) {
#ifndef BROKEN_EXCEPTIONS
    printf("%ld %ld %ld %ld %ld\n", 
	long(nQ), long(nR1), long(nR2), long(nR3), long(nR4));
    PyErr_SetString(PyExc_ValueError, "Q,R,wavelength have different lengths");
#endif
    return NULL;
  }
  magnetic_amplitude(nd,d,rho,mu,wavelength,P,expth,Aguide,nQ,Q,R1,R2,R3,R4);
  return Py_BuildValue("");
}


PyObject* Preflamp_rough(PyObject*obj,PyObject*args)
{
  PyObject *Q_obj,*R_obj,*d_obj,*rho_obj,*mu_obj,*sigma_obj, *wavelength_obj;
  const double *Q, *d, *rho, *mu, *sigma;
  double *wavelength;
  refl_complex *R;
  Py_ssize_t nQ, nR, nd, nrho, nmu, nsigma, nwavelength;

  if (!PyArg_ParseTuple(args, "OOOOOOO:reflrough", 
			&rho_obj,&mu_obj,&d_obj,&sigma_obj, &wavelength_obj,
			&Q_obj,&R_obj))
		 return NULL;
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(d_obj,d,nd);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(mu_obj,mu,nmu);
  INVECTOR(Q_obj,Q,nQ);
  INVECTOR(wavelength_obj, wavelength, nwavelength);
  OUTVECTOR(R_obj,R,nR);
  // interfaces should be one shorter than layers
  if (nd != nrho || nd != nmu || nd != nsigma+1) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,rho,mu,sigma have different lengths");
#endif
    return NULL;
  }
  if (nQ != nR || nwavelength != nQ) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "Q,R,wavelength have different lengths");
#endif
    return NULL;
  }
  reflrough_amplitude(nd,d,sigma,rho,mu, wavelength, nQ, Q, R);
  return Py_BuildValue("");
}



PyObject* Perf(PyObject*obj,PyObject*args)
{
  PyObject *data_obj, *result_obj;
  const double *data;
  double *result;
  int i;
  Py_ssize_t ndata, nresult;

  if (!PyArg_ParseTuple(args, "OO:erf", 
			&data_obj, &result_obj)) 
    return NULL;
  INVECTOR(data_obj,data, ndata);
  OUTVECTOR(result_obj, result, nresult);
  if (ndata != nresult) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "len(data) != nresult");
#endif
    return NULL;
  }
  for(i=0; i < ndata; i++)
    result[i] = erf(data[i]);
  return Py_BuildValue("");
}
