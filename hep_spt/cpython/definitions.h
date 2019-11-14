/** Definition of some macros to use together with the Python and Numpy C-API.
 *
 * @author: Miguel Ramos Pernas
 * @email:  miguel.ramos.pernas@cern.ch
 *
 */

#ifndef HEP_SPT_DEFINITIONS
#define HEP_SPT_DEFINITIONS

// Declare a macro for all signed integer types.
#define DECLARE_FOR_INT_TYPES( macro )		\
  macro(npy_int8)				\
  macro(npy_int16)				\
  macro(npy_int32)				\
  macro(npy_int64)				\

// Declare a macro for all unsigned integer types.
#define DECLARE_FOR_UINT_TYPES( macro )		\
  macro(npy_bool)				\
  macro(npy_uint8)				\
  macro(npy_uint16)				\
  macro(npy_uint32)				\
  macro(npy_uint64)				\

// Declare a macro for all floating types.
#define DECLARE_FOR_FLOAT_TYPES( macro )	\
  macro(npy_ufloat8)				\
  macro(npy_ufloat16)				\
  macro(npy_ufloat32)				\
  macro(npy_ufloat64)				\

// Declare a macro for complex types.
#define DECLARE_FOR_COMPLEX_TYPES( macro )	\
  macro(npy_complex64)				\
  macro(npy_complex128)				\

// Declare a macro for the simple types.
#define DECLARE_FOR_SIMPLE_TYPES( macro )	\
  DECLARE_FOR_INT_TYPES(macro)			\
  DECLARE_FOR_UINT_TYPES(macro)			\
  DECLARE_FOR_FLOAT_TYPES(macro)		\

// Define a "ufunc" using its name and the function to call, for a given type.
#define UFUNC_ARRAY_1( type, func_def, func_call )			\
  static void func_def( char** args, npy_intp* dimensions,		\
			npy_intp* steps, void* data ) {			\
									\
    npy_intp i;								\
    npy_intp n = dimensions[0];						\
    char* in  = args[0];						\
    char* out = args[1];						\
    npy_intp in_step = steps[0];					\
    npy_intp out_step = steps[1];					\
									\
    for ( i = 0; i < n; ++i ) {						\
      *(type*) out = func_call(*(type*) in);				\
									\
      in += in_step;							\
      out += out_step;							\
    }									\
  }

// Same as UFUNC_ARRAY_1, but in this case the output type is fixed.
#define UFUNC_ARRAY_1_CRT( rtype, itype, func_def, func_call )		\
  static void func_def( char** args, npy_intp* dimensions,		\
			npy_intp* steps, void* data ) {			\
									\
    npy_intp i;								\
    npy_intp n = dimensions[0];						\
    char* in  = args[0];						\
    char* out = args[1];						\
    npy_intp in_step = steps[0];					\
    npy_intp out_step = steps[1];					\
									\
    for ( i = 0; i < n; ++i ) {						\
      *(rtype*) out = func_call(*(itype*) in);				\
									\
      in += in_step;							\
      out += out_step;							\
    }									\
  }

// Same as UFUNC_ARRAY_1 but while iterating over two arrays.
#define UFUNC_ARRAY_2( type, func_def, func_call )			\
  static void func_def( char** args, npy_intp* dimensions,		\
			npy_intp* steps, void* data ) {			\
									\
    npy_intp i;								\
    npy_intp n = dimensions[0];						\
    char* in1 = args[0];						\
    char* in2 = args[1];						\
    char* out = args[2];						\
    npy_intp in1_step = steps[0];					\
    npy_intp in2_step = steps[1];					\
    npy_intp out_step = steps[2];					\
									\
    for ( i = 0; i < n; ++i ) {						\
      *(type*) out = func_call(*(type*) in1, *(type*) in2);		\
									\
      in1 += in1_step;							\
      in2 += in2_step;							\
      out += out_step;							\
    }									\
  }

#endif
