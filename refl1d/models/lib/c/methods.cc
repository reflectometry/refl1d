/* This program is public domain. */

#include <math.h>
#include <stdio.h>
#include <iostream>

//#define Py_LIMITED_API 0x03020000
#include <Python.h>

#include "reflcalc.h"
#include "methods.h"


#if defined(PY_VERSION_HEX) &&  (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif


#undef BROKEN_EXCEPTIONS


PyObject* Pcalculate_u1_u3(PyObject*obj,PyObject*args)
{
  PyObject *rhom_obj, *thetam_obj, *u1_obj, *u3_obj;
  Py_ssize_t nrhom, nthetam, nu1, nu3;
  double *rhom;
  const double *thetam;
  Cplx *u1, *u3;
  double H, Aguide;
  DECLARE_VECTORS(5);

  if (!PyArg_ParseTuple(args, "dOOdOO:calculate_u1_u3",
      &H, &rhom_obj, &thetam_obj, &Aguide, &u1_obj, &u3_obj))
    return NULL;
  INVECTOR(rhom_obj,rhom,nrhom);
  INVECTOR(thetam_obj,thetam,nthetam);
  OUTVECTOR(u1_obj,u1,nu1);
  OUTVECTOR(u3_obj,u3,nu3);
  if (nrhom != nthetam || nrhom != nu1 || nrhom != nu3) {
    //printf("%ld %ld %ld %ld %ld %ld\n",
    //    long(nd), long(nsigma), long(nrho), long(nirho), long(nrhom), long(nu1));
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "rhom,thetam,u1,u3 have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }

  for (Py_ssize_t i=0; i<nrhom; i++) {
    calculate_U1_U3(H, rhom[i], thetam[i], Aguide, u1[i], u3[i]);
    //sldb[i] = fabs(sldb[i]);
  }

  FREE_VECTORS();
  return Py_BuildValue("");
}



PyObject* Pmagnetic_amplitude(PyObject*obj,PyObject*args)
{
  PyObject *kz_obj,*rho_index_obj,*r1_obj,*r2_obj,*r3_obj,*r4_obj,
    *d_obj,*sigma_obj,*rho_obj,*irho_obj, *rhom_obj, *u1_obj, *u3_obj;
  Py_ssize_t nkz, nrho_index, nr1, nr2, nr3, nr4,
    nd, nsigma, nrho, nirho, nrhom, nu1, nu3;
  const double *kz, *d, *sigma, *rho, *irho, *rhom;
  const Cplx *u1, *u3;
  const int *rho_index;
  Cplx *r1, *r2, *r3, *r4;
  DECLARE_VECTORS(13);

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO:magnetic_amplitude",
      &d_obj, &sigma_obj,
      &rho_obj, &irho_obj, &rhom_obj,&u1_obj, &u3_obj,
      &kz_obj,&rho_index_obj,
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
    FREE_VECTORS();
    return NULL;
  }
  if (nkz != nr1 || nkz != nr2 || nkz != nr3 || nkz != nr4 || nkz != nrho_index) {
    //printf("%ld %ld %ld %ld %ld %ld\n",
    //    long(nkz), long(nr1), long(nr2), long(nr3), long(nr4), long(nrho_index));
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "kz,r1,r2,r3,r4,rho_index have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  magnetic_amplitude((int)nd, d, sigma, rho, irho, rhom, u1, u3,
                     (int)nkz, kz, rho_index, r1, r2, r3, r4);
  FREE_VECTORS();
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
  DECLARE_VECTORS(7);

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
    FREE_VECTORS();
    return NULL;
  }
  if (nrho < nd*nprofiles || nirho < nd*nprofiles) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "rho_index too high");
#endif
    FREE_VECTORS();
    return NULL;
  }
  if (nkz != nr || nrho_index != nkz) {
    //printf("%ld %ld %ld\n",
    //    long(nkz), long(nrho_index), long(nr));
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "kz,rho_index,r have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  reflectivity_amplitude((int)nd, d, sigma, rho, irho, (int)nkz, kz, rho_index, r);
  FREE_VECTORS();
  return Py_BuildValue("");
}

PyObject* Palign_magnetic(PyObject *obj, PyObject *args)
{
  PyObject *d_obj,*rho_obj,*irho_obj,*sigma_obj;
  PyObject *dM_obj,*rhoM_obj,*thetaM_obj,*sigmaM_obj;
  PyObject *output_obj;
  Py_ssize_t nd, nrho, nirho, nsigma;
  Py_ssize_t ndM, nrhoM, nthetaM, nsigmaM;
  Py_ssize_t noutput;
  double *d, *sigma, *rho, *irho;
  double *dM, *sigmaM, *rhoM, *thetaM;
  double *output;
  DECLARE_VECTORS(9);

  if (!PyArg_ParseTuple(args, "OOOOOOOOO:align_magnetic",
      &d_obj,&sigma_obj,&rho_obj,&irho_obj,
      &dM_obj,&sigmaM_obj,&rhoM_obj,&thetaM_obj,
      &output_obj))
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  INVECTOR(dM_obj,dM,ndM);
  INVECTOR(sigmaM_obj,sigmaM,nsigmaM);
  INVECTOR(rhoM_obj,rhoM,nrhoM);
  INVECTOR(thetaM_obj,thetaM,nthetaM);
  OUTVECTOR(output_obj,output,noutput);

  // interfaces should be one shorter than layers
  if (nd != nrho || nd != nirho || nd-1 != nsigma) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,sigma,rho,irho have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  if (ndM != nrhoM || ndM != nthetaM || ndM-1 != nsigmaM) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "dM,sigmaM,rhoM,thetaM have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }

  int newlen = align_magnetic((int)nd, d, sigma, rho, irho,
                              (int)ndM, dM, sigmaM, rhoM, thetaM,
                              ((int)noutput)/6, output);
  if (newlen < 0) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "output too short --- can be as large as 6x(#nuc+#mag)");
#endif
    FREE_VECTORS();
    return NULL;
  }
  FREE_VECTORS();
  return Py_BuildValue("i",newlen);
}

PyObject* Pcontract_by_area(PyObject*obj,PyObject*args)
{
  PyObject *d_obj,*rho_obj,*irho_obj,*sigma_obj;
  Py_ssize_t nd, nrho, nirho, nsigma;
  double *d, *sigma, *rho, *irho;
  double dA;
  DECLARE_VECTORS(4);

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
    FREE_VECTORS();
    return NULL;
  }
  int newlen = contract_by_area((int)nd, d, sigma, rho, irho, dA);
  FREE_VECTORS();
  return Py_BuildValue("i",newlen);
}

PyObject* Pcontract_mag(PyObject*obj,PyObject*args)
{
  PyObject *d_obj,*rho_obj,*irho_obj,*rhoM_obj,*thetaM_obj,*sigma_obj;
  Py_ssize_t nd, nrho, nirho, nrhoM, nthetaM, nsigma;
  double *d, *sigma, *rho, *irho, *rhoM, *thetaM;
  double dA;
  DECLARE_VECTORS(6);

  if (!PyArg_ParseTuple(args, "OOOOOOd:contract_mag",
      &d_obj,&sigma_obj,&rho_obj,&irho_obj,&rhoM_obj,&thetaM_obj,&dA))
    return NULL;
  INVECTOR(d_obj,d,nd);
  INVECTOR(sigma_obj,sigma,nsigma);
  INVECTOR(rho_obj,rho,nrho);
  INVECTOR(irho_obj,irho,nirho);
  INVECTOR(rhoM_obj,rhoM,nrhoM);
  INVECTOR(thetaM_obj,thetaM,nthetaM);
  // interfaces should be one shorter than layers
  if (nd != nrho || nd != nirho || nd != nrhoM || nd != nthetaM || nd != nsigma+1) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "d,rho,irho,rhoM,thetaM,sigma have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  int newlen = contract_mag((int)nd, d, sigma, rho, irho, rhoM, thetaM, dA);
  FREE_VECTORS();
  return Py_BuildValue("i",newlen);
}


PyObject* Pcontract_by_step(PyObject*obj,PyObject*args)
{
  PyObject *d_obj,*rho_obj,*irho_obj,*sigma_obj;
  Py_ssize_t nd, nrho, nirho, nsigma;
  double *d, *sigma, *rho, *irho;
  double dv;
  DECLARE_VECTORS(4);

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
    FREE_VECTORS();
    return NULL;
  }
  int newlen = contract_by_step((int)nd, d, sigma, rho, irho, dv);
  FREE_VECTORS();
  return Py_BuildValue("i",newlen);
}


PyObject* Pconvolve_gaussian(PyObject *obj, PyObject *args)
{
  PyObject *xi_obj,*yi_obj,*x_obj,*dx_obj,*y_obj;
  const double *xi, *yi, *x, *dx;
  double *y;
  Py_ssize_t nxi, nyi, nx, ndx, ny;
  DECLARE_VECTORS(5);

  if (!PyArg_ParseTuple(args, "OOOOO:convolve_gaussian",
			&xi_obj,&yi_obj,&x_obj,&dx_obj,&y_obj)) return NULL;
  INVECTOR(xi_obj,xi,nxi);
  INVECTOR(yi_obj,yi,nyi);
  INVECTOR(x_obj,x,nx);
  INVECTOR(dx_obj,dx,ndx);
  OUTVECTOR(y_obj,y,ny);
  if (nxi != nyi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_gaussian: xi and yi have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  if (nx != ndx || nx != ny) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_gaussian: x, dx and y have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  convolve_gaussian(nxi,xi,yi,nx,x,dx,y);
  FREE_VECTORS();
  return Py_BuildValue("");
}

PyObject* Pconvolve_uniform(PyObject *obj, PyObject *args)
{
  PyObject *xi_obj,*yi_obj,*x_obj,*dx_obj,*y_obj;
  const double *xi, *yi, *x, *dx;
  double *y;
  Py_ssize_t nxi, nyi, nx, ndx, ny;
  DECLARE_VECTORS(5);

  if (!PyArg_ParseTuple(args, "OOOOO:convolve_uniform",
			&xi_obj,&yi_obj,&x_obj,&dx_obj,&y_obj)) return NULL;
  INVECTOR(xi_obj,xi,nxi);
  INVECTOR(yi_obj,yi,nyi);
  INVECTOR(x_obj,x,nx);
  INVECTOR(dx_obj,dx,ndx);
  OUTVECTOR(y_obj,y,ny);
  if (nxi != nyi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_uniform: xi and yi have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  if (nx != ndx || nx != ny) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_uniform: x, dx and y have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  convolve_gaussian(nxi,xi,yi,nx,x,dx,y);
  FREE_VECTORS();
  return Py_BuildValue("");
}

PyObject* Pconvolve_sampled(PyObject *obj, PyObject *args)
{
  PyObject *xi_obj,*yi_obj,*xp_obj,*yp_obj,*x_obj,*dx_obj,*y_obj;
  const double *xi, *yi, *xp, *yp, *x, *dx;
  double *y;
  Py_ssize_t nxi, nyi, nxp, nyp, nx, ndx, ny;
  DECLARE_VECTORS(7);

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
    FREE_VECTORS();
    return NULL;
  }
  if (nxp != nyp) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: xp and yp have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  if (nx != ndx || nx != ny) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: x, dx and y have different lengths");
#endif
    FREE_VECTORS();
    return NULL;
  }
  convolve_sampled(nxi,xi,yi,nxp,xp,yp,nx,x,dx,y);
  FREE_VECTORS();
  return Py_BuildValue("");
}

PyObject* Pbuild_profile(PyObject *obj, PyObject *args)
{
  PyObject *z_obj,*offset_obj,*roughness_obj,*contrast_obj,*initial_value_obj,*profiles_obj;
  const double *z, *offset, *roughness, *contrast, *initial_value;
  double *profiles;
  Py_ssize_t nz, no, nr, nc, niv, np;
  DECLARE_VECTORS(6);

  if (!PyArg_ParseTuple(args, "OOOOOO:build_profile",
	   &z_obj,&offset_obj,&roughness_obj,&contrast_obj,&initial_value_obj,&profiles_obj)) return NULL;
  INVECTOR(z_obj,z,nz);
  INVECTOR(offset_obj,offset,no);
  INVECTOR(roughness_obj,roughness,nr);
  INVECTOR(contrast_obj,contrast,nc);
  INVECTOR(initial_value_obj,initial_value,niv);
  OUTVECTOR(profiles_obj,profiles,np);

  build_profile(nz, niv, no, z, offset, roughness, contrast, initial_value, profiles);
  FREE_VECTORS();
  return Py_BuildValue("");
}