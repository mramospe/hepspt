/** Definition of some macros to use together with the Python and Numpy C-API.
 *
 */


/// Check that the array has dimension no greater than one.
#ifndef CHECK_DIM_ARRAY
#define CHECK_DIM_ARRAY( arr )						\
  if ( PyArray_NDIM(arr) > 1 ) {					\
    PyErr_SetString(PyExc_TypeError,					\
		    "Number of dimensions in array must be zero or one"); \
    goto final;							\
  }
#endif


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
  CHECK_DIM_ARRAY(arr_a);						\
  CHECK_DIM_ARRAY(arr_b);						\

#endif
