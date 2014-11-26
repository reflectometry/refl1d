/* This program is public domain. */

#include <Python.h>
#include "reflcalc.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "methods.h"


#if defined(PY_VERSION_HEX) &&  (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif


#undef BROKEN_EXCEPTIONS


PyObject* Pmagnetic_amplitude(PyObject*obj,PyObject*args)
{
  PyObject *kz_obj,*rho_index_obj,*r1_obj,*r2_obj,*r3_obj,*r4_obj,
    *d_obj,*sigma_obj,*rho_obj,*irho_obj, *rhom_obj, *u1_obj, *u3_obj;
  Py_ssize_t nkz, nrho_index, nr1, nr2, nr3, nr4,
    nd, nsigma, nrho, nirho, nrhom, nu1, nu3;
  const double *kz, *d, *sigma, *rho, *irho, *rhom;
  const Cplx *u1, *u3;
  const int *rho_index;
  double Aguide;
  Cplx *r1, *r2, *r3, *r4;

  if (!PyArg_ParseTuple(args, "OOOOOOOdOOOOOO:magnetic_reflectivity",
      &d_obj, &sigma_obj,
      &rho_obj, &irho_obj, &rhom_obj,&u1_obj, &u3_obj,
      &Aguide,&kz_obj,&rho_index_obj,
      &r1_obj,&r2_obj,&r3_obj,&r4_obj))
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  INVECTOR(rhom_obj,rhom,nrhom);
  INVECTOR(u1_obj,u1,nu1);
  INVECTOR(u3_obj,u3,nu3);
  INVECTOR(kz_obj,kz,nkz);
  INVECTOR(rho_index_obj, rho_index, nrho_index);
  OUTVECTOR(r1_obj,r1,nr1);
  OUTVECTOR(r2_obj,r2,nr2);
  OUTVECTOR(r3_obj,r3,nr3);
  OUTVECTOR(r4_obj,r4,nr4);
  if (nd != nrho || nd != nirho || nd != nrhom || nd != nu1 || nd != nu3 || nd != nsigma+1) {
    //printf("%ld %ld %ld %ld %ld %ld\n",
    //    long(nd), long(nsigma), long(nrho), long(nirho), long(nrhom), long(nu1));
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,sigma,rho,irho,rhom,u1,u3 have different lengths");
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
  magnetic_amplitude((int)nd, d, sigma, rho, irho, rhom, u1, u3,
                     Aguide, (int)nkz, kz, rho_index, r1, r2, r3, r4);
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
    PyErr_SetString(PyExc_ValueError, "d,rho,irho,sigma have different lengths");
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
  reflectivity_amplitude((int)nd, d, sigma, rho, irho, (int)nkz, kz, rho_index, r);
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
  int newlen = contract_by_area((int)nd, d, sigma, rho, irho, dA);
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
  int newlen = contract_mag((int)nd, d, rho, irho, rhoM, thetaM, dA);
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
  int newlen = contract_by_step((int)nd, d, sigma, rho, irho, dv);
  return Py_BuildValue("i",newlen);
}


PyObject* Pconvolve(PyObject *obj, PyObject *args)
{
  PyObject *xi_obj,*yi_obj,*x_obj,*dx_obj,*y_obj;
  const double *xi, *yi, *x, *dx;
  double *y;
  Py_ssize_t nxi, nyi, nx, ndx, ny;

  if (!PyArg_ParseTuple(args, "OOOOO:convolve",
			&xi_obj,&yi_obj,&x_obj,&dx_obj,&y_obj)) return NULL;
  INVECTOR(xi_obj,xi,nxi);
  INVECTOR(yi_obj,yi,nyi);
  INVECTOR(x_obj,x,nx);
  INVECTOR(dx_obj,dx,ndx);
  OUTVECTOR(y_obj,y,ny);
  if (nxi != nyi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve: xi and yi have different lengths");
#endif
    return NULL;
  }
  if (nx != ndx || nx != ny) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve: x, dx and y have different lengths");
#endif
    return NULL;
  }
  convolve(nxi,xi,yi,nx,x,dx,y);
  return Py_BuildValue("");
}

PyObject* Pconvolve_sampled(PyObject *obj, PyObject *args)
{
  PyObject *xi_obj,*yi_obj,*xp_obj,*yp_obj,*x_obj,*dx_obj,*y_obj;
  const double *xi, *yi, *xp, *yp, *x, *dx;
  double *y;
  Py_ssize_t nxi, nyi, nxp, nyp, nx, ndx, ny;

  if (!PyArg_ParseTuple(args, "OOOOOOO:convolve_sampled",
	   &xi_obj,&yi_obj,&xp_obj,&yp_obj,&x_obj,&dx_obj,&y_obj)) return NULL;
  INVECTOR(xi_obj,xi,nxi);
  INVECTOR(yi_obj,yi,nyi);
  INVECTOR(xp_obj,xp,nxp);
  INVECTOR(yp_obj,yp,nyp);
  INVECTOR(x_obj,x,nx);
  INVECTOR(dx_obj,dx,ndx);
  OUTVECTOR(y_obj,y,ny);
  if (nxi != nyi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: xi and yi have different lengths");
#endif
    return NULL;
  }
  if (nxp != nyp) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: xp and yp have different lengths");
#endif
    return NULL;
  }
  if (nx != ndx || nx != ny) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: x, dx and y have different lengths");
#endif
    return NULL;
  }
  convolve_sampled(nxi,xi,yi,nxp,xp,yp,nx,x,dx,y);
  return Py_BuildValue("");
}


