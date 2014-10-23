#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// Kept as reference to prove that pointer tricks work
PyObject *Pcalc_g_zs_explicit(PyObject *self, PyObject *args)
{
    PyArrayObject *g_z_pao, *c_i_pao, *g_zs_pao;
    double lambda_0, lambda_1;
    double *g_z, *c_i, *g_zs;
    Py_ssize_t segments, layers, r, z;
    
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

    for (r=1; r<segments; ++r) {
    
        z=0;
        g_zs[z+layers*r] = (g_zs[z+layers*(r-1)] * lambda_0
            + g_zs[z+1+layers*(r-1)] * lambda_1
            + c_i[segments-r-1]) * g_z[z];
        
        for (z=1; z<(layers-1); ++z) {
            g_zs[z+layers*r] = (g_zs[z-1+layers*(r-1)] * lambda_1
                + g_zs[z+layers*(r-1)] * lambda_0
                + g_zs[z+1+layers*(r-1)] * lambda_1
                + c_i[segments-r-1]) * g_z[z];
        }
        
        g_zs[z+layers*r] = (g_zs[z+layers*(r-1)] * lambda_0
            + g_zs[z-1+layers*(r-1)] * lambda_1
            + c_i[segments-r-1]) * g_z[z];

    }   

    Py_RETURN_NONE;
}

// enhanced with pointer magic for a 40% speed increase
PyObject *Pcalc_g_zs(PyObject *self, PyObject *args)
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
    
    // note: at this point, accessing c_i or g_zs will segfault

    Py_RETURN_NONE;
}

// 12% speedup for uniform distributions or un-normalized (g_zs[:,0]=g_z[:])
PyObject *Pcalc_g_zs_uniform(PyObject *self, PyObject *args)
{
    PyArrayObject *g_z_pao, *g_zs_pao;
    double lambda_0, lambda_1;
    double *g_z, *g_zs, *g_zs_prev, *g_zs_prev_above, *g_zs_prev_below;
    Py_ssize_t segments, layers, s, z;
    
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

        //first and last layers have special cases
        *g_zs++ = (
                   (*g_zs_prev++) * lambda_0
                   + (*g_zs_prev_above++) * lambda_1
                   ) * (*g_z++);
        ++g_zs_prev_below; // didn't increment this one in the math

        for (z=1; z<(layers-1); ++z) {
            *g_zs++ = (
                       (*g_zs_prev_below++) * lambda_1
                       + (*g_zs_prev++) * lambda_0
                       + (*g_zs_prev_above++) * lambda_1
                       ) * (*g_z++);
        }

        //first and last layers have special cases
        *g_zs++ = (
                   (*g_zs_prev_below++) * lambda_1
                   + (*g_zs_prev++) * lambda_0
                   ) * (*g_z++);
        ++g_zs_prev_above; // didn't increment this one in the math

        //reset g_z
        g_z -= layers;

    }   
    
    // note: at this point, accessing c_i or g_zs will segfault

    Py_RETURN_NONE;
}


static PyMethodDef calc_g_zs_cex_methods[] =
{
    {"_calc_g_zs_explicit", Pcalc_g_zs_explicit, METH_VARARGS, 
    "_calc_g_zs_explicit(g_z,c_i,lambda_0,lambda_1,layers,segments): calculate G(z,s) using explicit array indexing"
    },
    {"_calc_g_zs", Pcalc_g_zs, METH_VARARGS, 
    "_calc_g_zs(g_z,c_i,lambda_0,lambda_1,layers,segments): calculate G(z,s) using pointer arithmatic optimizations"
    },
    {"_calc_g_zs_uniform", Pcalc_g_zs_uniform, METH_VARARGS, 
    "_calc_g_zs_uniform(g_z,c_i,lambda_0,lambda_1,layers,segments): calculate G(z,s) using pointer arithmatic optimizations"
    },
    {NULL, NULL, 0, NULL}
};

void initcalc_g_zs_cex(void)
{
    (void)Py_InitModule("calc_g_zs_cex", calc_g_zs_cex_methods);
    import_array();
}