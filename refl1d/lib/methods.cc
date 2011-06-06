/* This program is public domain. */

#include "reflcalc.h"
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


PyObject* Pmagnetic_amplitude(PyObject*obj,PyObject*args)
{
  PyObject *kz_obj,*rho_index_obj,*r1_obj,*r2_obj,*r3_obj,*r4_obj,
    *d_obj,*sigma_obj,*rho_obj,*irho_obj, *rhom_obj, *expth_obj;
  Py_ssize_t nkz, nrho_index, nr1, nr2, nr3, nr4,
    nd, nsigma, nrho, nirho, nrhom, nexpth;
  const double *kz, *d, *sigma, *rho, *irho, *rhom;
  const Cplx *expth;
  const int *rho_index;
  double Aguide;
  Cplx *r1, *r2, *r3, *r4;

  if (!PyArg_ParseTuple(args, "OOOOOOdOOOOOO:magnetic_reflectivity",
      &d_obj, &sigma_obj,
      &rho_obj, &irho_obj, &rhom_obj,&expth_obj,
      &Aguide,&kz_obj,&rho_index_obj,
      &r1_obj,&r2_obj,&r3_obj,&r4_obj))
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  INVECTOR(rhom_obj,rhom,nrhom);
  INVECTOR(expth_obj,expth,nexpth);
  INVECTOR(kz_obj,kz,nkz);
  INVECTOR(rho_index_obj, rho_index, nrho_index);
  OUTVECTOR(r1_obj,r1,nr1);
  OUTVECTOR(r2_obj,r2,nr2);
  OUTVECTOR(r3_obj,r3,nr3);
  OUTVECTOR(r4_obj,r4,nr4);
  if (nd != nrho || nd != nirho || nd != nrhom || nd != nexpth || nd != nsigma+1) {
    //printf("%ld %ld %ld %ld %ld %ld\n",
    //    long(nd), long(nsigma), long(nrho), long(nirho), long(nrhom), long(nexpth));
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,sigma,rho,irho,rhom,expth have different lengths");
#endif
    return NULL;
  }
  if (nkz != nr1 || nkz != nr2 || nkz != nr3 || nkz != nr4 || nkz != nrho_index) {
    //printf("%ld %ld %ld %ld %ld %ld\n",
    //    long(nkz), long(nr1), long(nr2), long(nr3), long(nr4), long(nrho_index));
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "kz,r1,r2,r3,r4,rho_index have different lengths");
#endif
    return NULL;
  }
  magnetic_amplitude(nd, d, sigma, rho, irho, rhom, expth,
                     Aguide, nkz, kz, rho_index, r1, r2, r3, r4);
  return Py_BuildValue("");
}


PyObject* Preflectivity_amplitude(PyObject*obj,PyObject*args)
{
  PyObject *kz_obj,*r_obj,*d_obj,*rho_obj,*irho_obj,*sigma_obj,*rho_index_obj;
  Py_ssize_t nkz, nr, nd, nrho, nirho, nsigma, nrho_index;
  const double *kz, *d, *sigma, *rho, *irho;
  const int *rho_index;
  int nprofiles;
  Cplx *r;

  if (!PyArg_ParseTuple(args, "OOOOOOO:reflectivity",
      &d_obj,&sigma_obj,&rho_obj,&irho_obj,
      &kz_obj,&rho_index_obj, &r_obj))
    return NULL;
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(d_obj,d,nd);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  INVECTOR(kz_obj,kz,nkz);
  INVECTOR(rho_index_obj, rho_index, nrho_index);
  OUTVECTOR(r_obj,r,nr);

  // Determine how many profiles we have
  nprofiles = 1;
  for (int i=0; i < nrho_index; i++)
    if (rho_index[i] > nprofiles-1) nprofiles = rho_index[i]+1;

  // interfaces should be one shorter than layers
  if (nrho%nd != 0 || nirho%nd != 0 || nd != nsigma+1) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,rho,mu,sigma have different lengths");
#endif
    return NULL;
  }
  if (nrho < nd*nprofiles || nirho < nd*nprofiles) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "rho_index too high");
#endif
    return NULL;
  }
  if (nkz != nr || nrho_index != nkz) {
    //printf("%ld %ld %ld\n",
    //    long(nkz), long(nrho_index), long(nr));
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "kz,rho_index,r have different lengths");
#endif
    return NULL;
  }
  reflectivity_amplitude(nd, d, sigma, rho, irho, nkz, kz, rho_index, r);
  return Py_BuildValue("");
}



PyObject* Pcontract_by_area(PyObject*obj,PyObject*args)
{
  PyObject *d_obj,*rho_obj,*irho_obj,*sigma_obj;
  Py_ssize_t nd, nrho, nirho, nsigma;
  double *d, *sigma, *rho, *irho;
  double dA;

  if (!PyArg_ParseTuple(args, "OOOOd:reflectivity",
      &d_obj,&sigma_obj,&rho_obj,&irho_obj,&dA))
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  // interfaces should be one shorter than layers
  if (nd != nrho || nd != nirho || nd != nsigma+1) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,rho,mu,sigma have different lengths");
#endif
    return NULL;
  }
  int newlen = contract_by_area(nd, d, sigma, rho, irho, dA);
  return Py_BuildValue("i",newlen);
}

PyObject* Pcontract_mag(PyObject*obj,PyObject*args)
{
  PyObject *d_obj,*rho_obj,*irho_obj,*rhoM_obj,*thetaM_obj;
  Py_ssize_t nd, nrho, nirho, nrhoM, nthetaM;
  double *d, *rho, *irho, *rhoM, *thetaM;
  double dA;

  if (!PyArg_ParseTuple(args, "OOOOOd:contract_mag",
      &d_obj,&rho_obj,&irho_obj,&rhoM_obj,&thetaM_obj,&dA))
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  INVECTOR(rhoM_obj,rhoM,nrhoM);
  INVECTOR(thetaM_obj,thetaM,nthetaM);
  // interfaces should be one shorter than layers
  if (nd != nrho || nd != nirho || nd != nrhoM || nd != nthetaM) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,rho,irho,rhoM,thetaM have different lengths");
#endif
    return NULL;
  }
  int newlen = contract_mag(nd, d, rho, irho, rhoM, thetaM, dA);
  return Py_BuildValue("i",newlen);
}


PyObject* Pcontract_by_step(PyObject*obj,PyObject*args)
{
  PyObject *d_obj,*rho_obj,*irho_obj,*sigma_obj;
  Py_ssize_t nd, nrho, nirho, nsigma;
  double *d, *sigma, *rho, *irho;
  double dv;

  if (!PyArg_ParseTuple(args, "OOOOOd:reflectivity",
      &d_obj,&sigma_obj,&rho_obj,&irho_obj,&dv))
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  // interfaces should be one shorter than layers
  if (nd != nrho || nd != nirho || nd != nsigma+1) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,rho,mu,sigma have different lengths");
#endif
    return NULL;
  }
  int newlen = contract_by_step(nd, d, sigma, rho, irho, dv);
  return Py_BuildValue("i",newlen);
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
