#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *Pcalc_g_zs_inner(PyObject *self, PyObject *args)
{
    PyArrayObject *g_z_pao, *c_i_pao, *g_zs_pao;
    double lambda_0, lambda_1;
    double *g_z, *c_i, *g_zs;
    Py_ssize_t segments, layers, r, z;
    

    // .. for reference
    if (!PyArg_ParseTuple(args, "O!O!O!ddnn",&PyArray_Type,&g_z_pao,
            &PyArray_Type,&c_i_pao, &PyArray_Type,&g_zs_pao, 
            &lambda_0, &lambda_1, &layers, &segments))
    {return NULL;}

    g_z=(double *) PyArray_DATA(g_z_pao);
    c_i=(double *) PyArray_DATA(c_i_pao);
    g_zs=(double *) PyArray_DATA(g_zs_pao);

    for (r=1; r<segments; r++) {
    
        z=0;
        g_zs[z+layers*r] = (g_zs[z+layers*(r-1)] * lambda_0
            + g_zs[z+1+layers*(r-1)] * lambda_1
            + c_i[segments-r-1]) * g_z[z];
        
        for (z=1; z<(layers-1); z++) {
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


static PyMethodDef calc_g_zs_cex_methods[] =
{
    {"_calc_g_zs_inner", Pcalc_g_zs_inner, METH_VARARGS, 
    "_calc_g_zs_inner(g_z,c_i,lambda_0,lambda_1,layers,segments): calculate G(z,s)"
    },
    {NULL, NULL, 0, NULL}
};

void initcalc_g_zs_cex(void)
{
    (void)Py_InitModule("calc_g_zs_cex", calc_g_zs_cex_methods);
    import_array();
}