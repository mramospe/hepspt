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
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

// STD
#include <math.h>

// Local
#include "definitions.h"


/** Calculate the bit length of the values in an object.
 *
 * It is equivalent to the number of bits necessary to represent a number.
 */
#define BIT_LENGTH( type )						\
									\
  type _##type##_bit_length( const type i ) {				\
									\
    type c = i / 2;							\
    type r = i % 2;							\
    type l = (c != 0 || r != 0);					\
									\
    if ( c != 0 )							\
      l += _##type##_bit_length(c);					\
									\
    return l;								\
  }									\
  UFUNC_ARRAY_1(type, type##_bit_length, _##type##_bit_length);	\

DECLARE_FOR_INT_TYPES(BIT_LENGTH)
DECLARE_FOR_UINT_TYPES(BIT_LENGTH)

PyUFuncGenericFunction bit_length_funcs[9] = {&npy_bool_bit_length,
					      &npy_int8_bit_length,
					      &npy_int16_bit_length,
					      &npy_int32_bit_length,
					      &npy_int64_bit_length,
					      &npy_uint8_bit_length,
					      &npy_uint16_bit_length,
					      &npy_uint32_bit_length,
					      &npy_uint64_bit_length};

static void* bit_length_data[9] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

static char bit_length_types[18] = {NPY_BOOL, NPY_BOOL,
				    NPY_INT8, NPY_INT8,
				    NPY_INT16, NPY_INT16,
				    NPY_INT32, NPY_INT32,
				    NPY_INT64, NPY_INT64,
				    NPY_UINT8, NPY_UINT8,
				    NPY_UINT16, NPY_UINT16,
				    NPY_UINT32, NPY_UINT32,
				    NPY_UINT64, NPY_UINT64};



/** Calculate the greatest common divisor of two numbers.
 *
 *  This function always returns positive numbers, although
 *  "a" and "b" can be positive or negative.
 */
#define GCD( type )					\
  type _##type##_gcd( type a, type b ) {		\
							\
    while ( b ) {					\
							\
      const type r = a % b;				\
							\
      a = b;						\
      b = r;						\
    }							\
							\
    return abs(a);					\
  }							\
  UFUNC_ARRAY_2(type, type##_gcd, _##type##_gcd);	\

DECLARE_FOR_INT_TYPES(GCD)
DECLARE_FOR_UINT_TYPES(GCD)

PyUFuncGenericFunction gcd_funcs[9] = {&npy_bool_gcd,
				       &npy_int8_gcd,
				       &npy_int16_gcd,
				       &npy_int32_gcd,
				       &npy_int64_gcd,
				       &npy_uint8_gcd,
				       &npy_uint16_gcd,
				       &npy_uint32_gcd,
				       &npy_uint64_gcd};

static void* gcd_data[9] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

static char gcd_types[27] = {NPY_BOOL, NPY_BOOL, NPY_BOOL,
			     NPY_INT8, NPY_INT8, NPY_INT8,
			     NPY_INT16, NPY_INT16, NPY_INT16,
			     NPY_INT32, NPY_INT32, NPY_INT32,
			     NPY_INT64, NPY_INT64, NPY_INT64,
			     NPY_UINT8, NPY_UINT8, NPY_UINT8,
			     NPY_UINT16, NPY_UINT16, NPY_UINT16,
			     NPY_UINT32, NPY_UINT32, NPY_UINT32,
			     NPY_UINT64, NPY_UINT64, NPY_UINT64};


/** Calculate the binary representation (as an integer) of the values in an array.
 *
 * The object is converted to an array if necessary.
 */
#define IBINARY_REPR( type )								\
											\
  npy_int64 _##type##_ibinary_repr( const type i ) {					\
											\
    type      c = i / 2;								\
    npy_int64 r = i % 2;								\
											\
    if ( c != 0 )									\
      r += 10*_##type##_ibinary_repr(c);						\
											\
    return r;										\
  }											\
  UFUNC_ARRAY_1_CRT(npy_int64, type, type##_ibinary_repr, _##type##_ibinary_repr); 	\

DECLARE_FOR_INT_TYPES(IBINARY_REPR)
DECLARE_FOR_UINT_TYPES(IBINARY_REPR)

PyUFuncGenericFunction ibinary_repr_funcs[9] = {&npy_bool_ibinary_repr,
						&npy_int8_ibinary_repr,
						&npy_int16_ibinary_repr,
						&npy_int32_ibinary_repr,
						&npy_int64_ibinary_repr,
						&npy_uint8_ibinary_repr,
						&npy_uint16_ibinary_repr,
						&npy_uint32_ibinary_repr,
						&npy_uint64_ibinary_repr};

static void* ibinary_repr_data[9] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

static char ibinary_repr_types[18] = {NPY_BOOL, NPY_INT64,
				      NPY_INT8, NPY_INT64,
				      NPY_INT16, NPY_INT64,
				      NPY_INT32, NPY_INT64,
				      NPY_INT64, NPY_INT64,
				      NPY_UINT8, NPY_INT64,
				      NPY_UINT16, NPY_INT64,
				      NPY_UINT32, NPY_INT64,
				      NPY_UINT64, NPY_INT64};


// No methods are exported, since NumPy handles the "ufunc" objects.
static PyMethodDef Methods[] = {
  {NULL, NULL, 0, NULL}
};


/** Definition of the module.
 *
 */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef math_aux_cpy = {
  PyModuleDef_HEAD_INIT,
  "math_aux_cpy",
  "CPython functions for the 'math_aux' module.",
  -1,
  Methods,
  NULL,
  NULL,
  NULL,
  NULL
};
#endif


/** Function to initialize the module.
 *
 */
#if PY_MAJOR_VERSION >= 3

PyMODINIT_FUNC PyInit_math_aux_cpy( void ) {

#define INITERROR return NULL

#else

  void initmath_aux_cpy( void ) {

#define INITERROR return

#endif

#if PY_MAJOR_VERSION >= 3
    PyObject* module = PyModule_Create(&math_aux_cpy);
#else
    PyObject* module = Py_InitModule("math_aux_cpy", Methods);
#endif

    if ( module == NULL )
      INITERROR;

    import_array();
    import_umath();

    PyObject* dict = PyModule_GetDict(module);

    PyObject* bit_length = PyUFunc_FromFuncAndData(bit_length_funcs, bit_length_data, bit_length_types, 9, 1, 1,
						   PyUFunc_None, "bit_length",
						   "Calculate the bit length of the values in an object.",
						   0);
    PyDict_SetItemString(dict, "bit_length", bit_length);
    Py_DECREF(bit_length);

    PyObject* gcd = PyUFunc_FromFuncAndData(gcd_funcs, gcd_data, gcd_types, 9, 2, 1,
					    PyUFunc_None, "gcd",
					    "Calculate the greatest common divisor of two numbers.",
					    0);
    PyDict_SetItemString(dict, "gcd", gcd);
    Py_DECREF(gcd);

    PyObject* ibinary_repr = PyUFunc_FromFuncAndData(ibinary_repr_funcs, ibinary_repr_data, ibinary_repr_types, 9, 1, 1,
						     PyUFunc_None, "ibinary_repr",
						     "Calculate the binary representation (as an integer) of the values in an array.",
						     0);
    PyDict_SetItemString(dict, "ibinary_repr", ibinary_repr);
    Py_DECREF(ibinary_repr);

    // Set the value of "__all__" to an empty list
    PyObject* l = PyList_New(0);
    PyDict_SetItemString(dict, "__all__", l);
    Py_DECREF(l);

#if PY_MAJOR_VERSION >= 3

    return module;

#endif
  }
