/* This program is public domain. */

#include <Python.h>

// Vector binding glue
#define REF(obj,name  ) PyObject_GetAttrString(obj,#name)
#define SET(obj,name,v) PyObject_SetAttrString(obj,#name,v)

#define INVECTOR(obj,buf,len)										\
    do { \
        int err = PyObject_AsReadBuffer(obj, (const void **)(&buf), &len); \
        if (err < 0) return NULL; \
        len /= sizeof(*buf); \
    } while (0)

#define OUTVECTOR(obj,buf,len) \
    do { \
        int err = PyObject_AsWriteBuffer(obj, (void **)(&buf), &len); \
        if (err < 0) return NULL; \
        len /= sizeof(*buf); \
    } while (0)
#define SCALAR(obj) PyFloat_AsDouble(obj)

PyObject* pyvector(int n, double v[]);

PyObject* Perf(PyObject*obj,PyObject*args);
PyObject* Preflectivity_amplitude(PyObject*obj,PyObject*args);
PyObject* Pmagnetic_amplitude(PyObject* obj, PyObject* args);
PyObject* Pconvolve(PyObject*obj,PyObject*args);
PyObject* Pcontract_by_step(PyObject*obj,PyObject*args);
PyObject* Pcontract_by_area(PyObject*obj,PyObject*args);
