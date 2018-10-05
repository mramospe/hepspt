/** Functions to boost certain calculations involving numpy.ndarray objects.
 *
 * These functions profit from the Python and Numpy C-API.
 *
 * @author: Miguel Ramos Pernas
 * @email:  miguel.ramos.pernas@cern.ch
 *
 */

// C-API
#include <Python.h>
#include "numpy/arrayobject.h"

// STD
#include <math.h>

// Local
#include "definitions.h"
#include "types.h"


/// Send an error saying that values have been found outside some bounds
#ifndef OUT_OF_BOUNDS_ERROR
#define OUT_OF_BOUNDS_ERROR {						\
    PyErr_SetString(PyExc_ValueError, "Found values lying outside the provided edges"); \
    goto final;								\
  }									\

#endif


/** Assign a weight to the values in an input array using a set of edges.
 *
 */
static PyObject* weights_by_edges( PyObject *self, PyObject *args, PyObject *kwds ) {

  PyObject *in_values;
  PyObject *in_edges;
  PyObject *in_weights;

  static char* kwlist[] = {"values", "edges", "weights", NULL};
  if ( !PyArg_ParseTupleAndKeywords(args, kwds, "OOO:weights_by_edges", kwlist, &in_values, &in_edges, &in_weights) )
    goto final;

  PyArrayObject* values  = (PyArrayObject*) PyArray_FROM_O(in_values);
  PyArrayObject* edges   = (PyArrayObject*) PyArray_FROM_O(in_edges);
  PyArrayObject* weights = (PyArrayObject*) PyArray_FROM_O(in_weights);

  CHECK_ARRAY_1D(values);
  CHECK_ARRAY_1D(edges);
  CHECK_ARRAY_1D(weights);

  if ( PyArray_SIZE(weights) + 1 != PyArray_SIZE(edges) ) {
    PyErr_SetString(PyExc_TypeError, "Edges must have length equal to that from the array of weights plus one");
    goto final;
  }

  PyArrayObject* output = (PyArrayObject*) PyArray_NewLikeArray(values, NPY_ANYORDER, PyArray_DESCR(weights), 1);

  PyObject* iv = PyArray_IterNew((PyObject*) values);
  PyObject* io = PyArray_IterNew((PyObject*) output);

  while ( PyArray_ITER_NOTDONE(iv) ) {

    // Access to the input/output data
    npy_double* v_dt = (npy_double*) PyArray_ITER_DATA(iv);
    npy_double* o_dt = (npy_double*) PyArray_ITER_DATA(io);

    // (Re-)initialize the iterators
    PyObject* iep = PyArray_IterNew((PyObject*) edges);
    PyObject* ien = PyArray_IterNew((PyObject*) edges);
    PyObject* iw  = PyArray_IterNew((PyObject*) weights);

    int vtype = PyArray_TYPE(values);
    int etype = PyArray_TYPE(edges);
    int wtype = PyArray_TYPE(weights);

    void* ep_dt = PyArray_ITER_DATA(iep);
    void* en_dt = PyArray_ITER_DATA(ien);
    void* w_dt  = PyArray_ITER_DATA(iw);

    if ( ARRAY_LT(vtype, v_dt, etype, ep_dt) )
      OUT_OF_BOUNDS_ERROR;

    // Advance one step to define the "next" element of the edges.
    PyArray_ITER_NEXT(ien);

    while ( PyArray_ITER_NOTDONE(ien) ) {

      w_dt  = PyArray_ITER_DATA(iw);
      ep_dt = PyArray_ITER_DATA(iep);
      en_dt = PyArray_ITER_DATA(ien);

      if ( ARRAY_GE(vtype, v_dt, etype, ep_dt) && ARRAY_LT(vtype, v_dt, etype, en_dt) )
	ARRAY_ASSIGN(wtype, o_dt, wtype, w_dt);

      PyArray_ITER_NEXT(iep);
      PyArray_ITER_NEXT(ien);
      PyArray_ITER_NEXT(iw);
    }

    // Accept the value for the last bin if it lies in the rightmost edge
    if ( ARRAY_EQ(vtype, v_dt, etype, en_dt) )
	ARRAY_ASSIGN(wtype, o_dt, wtype, w_dt);

    if ( ARRAY_GT(vtype, v_dt, etype, en_dt) )
      OUT_OF_BOUNDS_ERROR;

    Py_DECREF(iep);
    Py_DECREF(ien);
    Py_DECREF(iw);

    PyArray_ITER_NEXT(iv);
    PyArray_ITER_NEXT(io);
  }

  Py_DECREF(iv);
  Py_DECREF(io);

  return (PyObject*) output;

 final:
  Py_XDECREF(values);
  Py_XDECREF(edges);
  Py_XDECREF(weights);
  return NULL;
}


/** Definition of the functions to be exported.
 *
 */
PyMethodDef Methods[] = {

  {"weights_by_edges", (PyCFunction) weights_by_edges, METH_VARARGS|METH_KEYWORDS,
   "Assign a weight to the values in an input array using a set of edges."},

  {NULL, NULL, 0, NULL}
};


/** Definition of the module.
 *
 */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef histograms_cpy = {
  PyModuleDef_HEAD_INIT,
  "histograms_cpy",
  "CPython functions for the 'histograms' module.",
  -1,
  Methods
};
#endif


/** Function to initialize the module.
 *
 */
#if PY_MAJOR_VERSION >= 3

PyMODINIT_FUNC PyInit_histograms_cpy( void ) {

#define INITERROR return NULL

#else

void inithistograms_cpy( void ) {

#define INITERROR return

#endif

  import_array();

#if PY_MAJOR_VERSION >= 3
  PyObject* module = PyModule_Create(&histograms_cpy);
#else
  PyObject* module = Py_InitModule("histograms_cpy", Methods);
#endif

  if ( module == NULL )
    INITERROR;

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}
