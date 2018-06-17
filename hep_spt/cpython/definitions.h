/** Definition of some macros to use together with the Python and Numpy C-API.
 *
 * @author: Miguel Ramos Pernas
 * @email:  miguel.ramos.pernas@cern.ch
 *
 */


/// Check the dimension of two arrays, which must coincide and be no greater than one.
#ifndef CHECK_DIM_ARRAYS
#define CHECK_DIM_ARRAYS( arr_a, arr_b )				\
									\
  if ( PyArray_NDIM(arr_a) != PyArray_NDIM(arr_b) ) {			\
    PyErr_SetString(PyExc_TypeError,					\
		    "Number of dimensions of the arrays must coincide"); \
    goto final;								\
  }									\
									\
  if ( !PyArray_CompareLists(PyArray_SHAPE(arr_a),			\
			     PyArray_SHAPE(arr_b),			\
			     PyArray_NDIM(arr_a)) ) {			\
    PyErr_SetString(PyExc_TypeError,					\
		    "Shape of the arrays must coincide");		\
    goto final;								\
  }									\

#endif


#ifndef CHECK_INT_ARRAY
#define CHECK_INT_ARRAY( arr )						\
  if ( !PyArray_ISINTEGER(arr) ) {					\
    PyErr_SetString(PyExc_TypeError, "Only integer numbers are allowed"); \
    goto final;								\
  }									\

#endif
