// Can't use Py_LIMITED_API with numpy arrays
//#define Py_LIMITED_API 0x03020000
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>


PyObject *Pcalc_g_zs(PyObject *self, PyObject *args)
{
    PyArrayObject *g_z_pao, *c_i_pao, *g_zs_pao;
    double lambda_0, lambda_1, cval;
    double *g_z, *c_i, *g_zs, *g_zs_next;
    Py_ssize_t segments, layers, r, z, i;
    
    if (!PyArg_ParseTuple(args, "O!O!O!ddnn",
                            &PyArray_Type,&g_z_pao,
                            &PyArray_Type,&c_i_pao,
                            &PyArray_Type,&g_zs_pao, 
                            &lambda_0,
                            &lambda_1,
                            &layers,
                            &segments))
    {return NULL;}

    g_z=(double *) PyArray_DATA(g_z_pao);
    c_i=(double *) PyArray_DATA(c_i_pao);
    g_zs=(double *) PyArray_DATA(g_zs_pao);
    g_zs_next = g_zs + layers;

    for (r=1; r<segments; ++r) {
        cval = c_i[segments-r-1];
        i = layers*(r-1); // Precalculating this index gave a 30% speedup

        //g_zs[i+layers] = ... switching to a pointer accounted for 14%
        *g_zs_next++ = g_z[0] * (
            g_zs[i] * lambda_0
            + g_zs[1+i] * lambda_1
            + cval
        );

        for (z=1; z<(layers-1); ++z) {
            *g_zs_next++ = g_z[z] * (
                g_zs[z+i-1] * lambda_1
                + g_zs[z+i] * lambda_0
                + g_zs[z+i+1] * lambda_1
                + cval
            );
        }

        *g_zs_next++ = g_z[z] * (
            g_zs[z+i] * lambda_0
            + g_zs[z+i-1] * lambda_1
            + cval
        );

    }   

    Py_RETURN_NONE;
}

// 15% speedup for uniform distributions
PyObject *Pcalc_g_zs_uniform(PyObject *self, PyObject *args)
{
    PyArrayObject *g_z_pao, *g_zs_pao;
    double lambda_0, lambda_1;
    double *g_z, *g_zs, *g_zs_next;
    Py_ssize_t segments, layers, r, z, i;
    
    if (!PyArg_ParseTuple(args, "O!O!ddnn",
                            &PyArray_Type,&g_z_pao,
                            &PyArray_Type,&g_zs_pao,
                            &lambda_0,
                            &lambda_1,
                            &layers,
                            &segments))
    {return NULL;}

    g_z=(double *) PyArray_DATA(g_z_pao);
    g_zs=(double *) PyArray_DATA(g_zs_pao);
    g_zs_next = g_zs + layers;

    for (r=1; r<segments; ++r) {
        i = layers*(r-1); // Precalculating this index gave a 30% speedup

        //g_zs[i+layers] = ...
        *g_zs_next++ = g_z[0] * (
            g_zs[i] * lambda_0
            + g_zs[1+i] * lambda_1
        );

        for (z=1; z<(layers-1); ++z) {
            *g_zs_next++ = g_z[z] * (
                g_zs[z+i-1] * lambda_1
                + g_zs[z+i] * lambda_0
                + g_zs[z+i+1] * lambda_1
            );
        }

        *g_zs_next++ = g_z[z] * (
            g_zs[z+i] * lambda_0
            + g_zs[z+i-1] * lambda_1
        );

    }   

    Py_RETURN_NONE;
}

// enhanced with pointer magic for a 20% speed decrease?!
PyObject *Pcalc_g_zs_pointers(PyObject *self, PyObject *args)
{
    PyArrayObject *g_z_pao, *c_i_pao, *g_zs_pao;
    double lambda_0, lambda_1, cval;
    double *g_z, *c_i, *g_zs, *g_zs_prev, *g_zs_prev_above, *g_zs_prev_below;
    Py_ssize_t segments, layers, s, z;
    
    if (!PyArg_ParseTuple(args, "O!O!O!ddnn",
                            &PyArray_Type,&g_z_pao,
                            &PyArray_Type,&c_i_pao,
                            &PyArray_Type,&g_zs_pao, 
                            &lambda_0,
                            &lambda_1,
                            &layers,
                            &segments))
    {return NULL;}


    g_z=(double *) PyArray_DATA(g_z_pao);
    c_i=(double *) PyArray_DATA(c_i_pao)+segments-1; // iterate backwards
    g_zs=(double *) PyArray_DATA(g_zs_pao);
    
    // Skip the first layer, which was defined outside this function    
    g_zs_prev = g_zs;
    g_zs_prev_above = g_zs + 1;
    g_zs_prev_below = g_zs - 1;
    g_zs += layers;

    /*
        for pointer voodoo, keep in mind that
        *ptr++
        is the same as
        *(ptr++)
        meaning, "Pointer moves to the next position (of an array),
        but returns the old content"
    */ 

    for (s=1; s<segments; ++s) {
        cval = *--c_i; // decrement and use the NEW position's content

        //first and last layers have special cases
        *g_zs++ = (
                   (*g_zs_prev++) * lambda_0
                   + (*g_zs_prev_above++) * lambda_1
                   + cval
                   ) * (*g_z++);
        ++g_zs_prev_below; // didn't increment this one in the math

        for (z=1; z<(layers-1); ++z) {
            *g_zs++ = (
                       (*g_zs_prev_below++) * lambda_1
                       + (*g_zs_prev++) * lambda_0
                       + (*g_zs_prev_above++) * lambda_1
                       + cval
                       ) * (*g_z++);
        }

        //first and last layers have special cases
        *g_zs++ = (
                   (*g_zs_prev_below++) * lambda_1
                   + (*g_zs_prev++) * lambda_0
                   + cval
                   ) * (*g_z++);
        ++g_zs_prev_above; // didn't increment this one in the math

        //reset g_z
        g_z -= layers;

    }   
    
    // note: at this point, accessing g_zs could segfault

    Py_RETURN_NONE;
}


static PyMethodDef calc_g_zs_cex_methods[] =
{
    {"_calc_g_zs", Pcalc_g_zs, METH_VARARGS, 
    "_calc_g_zs(g_z,c_i,g_zs,lambda_0,lambda_1,layers,segments): calculate G(z,s)"
    },
    {"_calc_g_zs_uniform", Pcalc_g_zs_uniform, METH_VARARGS, 
    "_calc_g_zs_uniform(g_z,g_zs,lambda_0,lambda_1,layers,segments): calculate G(z,s) for uniform chains"
    },
    {"_calc_g_zs_pointers", Pcalc_g_zs_pointers, METH_VARARGS, 
    "_calc_g_zs_pointers(g_z,c_i,g_zs,lambda_0,lambda_1,layers,segments): calculate G(z,s) using pointer tricks"
    },
    {NULL, NULL, 0, NULL}
};

/*
void initcalc_g_zs_cex(void)
{
    (void)Py_InitModule("calc_g_zs_cex", calc_g_zs_cex_methods);
    import_array();
}
*/

#define MODULE_DOC "calc_g_zs C Extension"
#define MODULE_NAME "calc_g_zs_cex"
#define MODULE_INIT2 initcalc_g_zs_cex
#define MODULE_INIT3 PyInit_calc_g_zs_cex
#define MODULE_METHODS calc_g_zs_cex_methods

/* ==== boilerplate python 2/3 interface bootstrap ==== */

#if defined(WIN32) && !defined(__MINGW32__)
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

#if PY_MAJOR_VERSION >= 3

  DLL_EXPORT PyMODINIT_FUNC MODULE_INIT3(void)
  {
    PyObject* blah;
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
    blah = PyModule_Create(&moduledef);
    import_array();
    return blah;
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
    import_array();
  }

#endif /* PY_MAJOR_VERSION >= 3 */
