#ifndef __FUNCTION_TYPES__
#define __FUNCTION_TYPES__

#define TYPED_OPERATION( oper, name, type, arr )			\
  switch ( type ) {							\
  case NPY_BOOL:							\
    {									\
      npy_bool* name = (npy_bool*) arr;					\
      oper;								\
      break;								\
    }									\
  case NPY_INT8:							\
    {									\
      npy_int8* name = (npy_int8*) arr;					\
      oper;								\
      break;								\
    }									\
  case NPY_INT16:							\
    {									\
      npy_int16* name = (npy_int16*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_INT32:							\
    {									\
      npy_int32* name = (npy_int32*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_INT64:							\
    {									\
      npy_int64* name = (npy_int64*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_UINT8:							\
    {									\
      npy_uint8* name = (npy_uint8*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_UINT16:							\
    {									\
      npy_uint16* name = (npy_uint16*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_UINT32:							\
    {									\
      npy_uint32* name = (npy_uint32*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_UINT64:							\
    {									\
      npy_uint64* name = (npy_uint64*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_FLOAT16:							\
    {									\
      npy_float16* name = (npy_float16*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_FLOAT32:							\
    {									\
      npy_float32* name = (npy_float32*) arr;				\
      oper;								\
      break;								\
    }									\
  case NPY_FLOAT64:							\
    {									\
      npy_float64* name = (npy_float64*) arr;				\
      oper;								\
      break;								\
    }									\
  default:								\
    {									\
      PyErr_SetString(PyExc_TypeError, "Input array data type is not allowed");	\
      break;								\
    }									\
  }

/*****************************************/
/********** Assignment operator **********/
/*****************************************/

#define ASSIGN (*output) = (*input)

inline void ARRAY_ASSIGN( int type_o, void* o, int type_i, int i ) {
  TYPED_OPERATION(TYPED_OPERATION(ASSIGN, output, type_o, o),
		  input, type_i, i);
}

/****************************************/
/********* Arithmetic operators *********/
/****************************************/

#define ADD (*output) = (*first) + (*second)
#define SUB (*output) = (*first) - (*second)
#define MUL (*output) = (*first) * (*second)
#define DIV (*output) = (*first) / (*second)

#define DEFINE_ARITHMETIC_FUNCTION( oper )				\
  void ARRAY_##oper( int type_o, void* o, int type_a, void* a, int type_b, void* b ) { \
    TYPED_OPERATION(TYPED_OPERATION(TYPED_OPERATION(oper, output, type_o, o), \
				    first, type_a, a),			\
		    second, type_b, b);					\
  }

#define DEFINE_ARITHMETIC_INPLACE_FUNCTION( oper )			\
  void ARRAY_INPLACE_##oper( int type_o, void* o, int type_a, void* a, int type_b, void* b ) { \
    TYPED_OPERATION(TYPED_OPERATION(TYPED_OPERATION(oper, output, type_a, a), \
				    first, type_a, a),			\
		    second, type_b, b);					\
  }

DEFINE_ARITHMETIC_FUNCTION(ADD);
DEFINE_ARITHMETIC_FUNCTION(SUB);
DEFINE_ARITHMETIC_FUNCTION(MUL);
DEFINE_ARITHMETIC_FUNCTION(DIV);

DEFINE_ARITHMETIC_INPLACE_FUNCTION(ADD);
DEFINE_ARITHMETIC_INPLACE_FUNCTION(SUB);
DEFINE_ARITHMETIC_INPLACE_FUNCTION(MUL);
DEFINE_ARITHMETIC_INPLACE_FUNCTION(DIV);

/*****************************************/
/*********** Boolean operators ***********/
/*****************************************/

#define LT output = (*first) < (*second)
#define LE output = (*first) <= (*second)
#define EQ output = (*first) == (*second)
#define GE output = (*first) >= (*second)
#define GT output = (*first) > (*second)

#define DEFINE_BOOLEAN_BINARY_FUNCTION( oper )				\
  npy_bool ARRAY_##oper( int type_a, void* a, int type_b, void* b ) {	\
    npy_bool output;							\
    TYPED_OPERATION(TYPED_OPERATION(oper, first, type_a, a), second, type_b, b)	\
      return output;							\
  }

// Define the boolean binary functions
DEFINE_BOOLEAN_BINARY_FUNCTION(LT);
DEFINE_BOOLEAN_BINARY_FUNCTION(LE);
DEFINE_BOOLEAN_BINARY_FUNCTION(EQ);
DEFINE_BOOLEAN_BINARY_FUNCTION(GE);
DEFINE_BOOLEAN_BINARY_FUNCTION(GT);

#endif
