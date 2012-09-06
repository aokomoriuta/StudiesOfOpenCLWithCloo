#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

//! REAL is provided by compiler option
typedef REAL Real;

//! Multiply matrix by vector
/*!
	\param result vector which result is stored to
	\param left multiplied vector
	\param right multiplying vector
*/
__kernel void Matrix_x_Vector(
	__global Real* result,
	const __global Real* matrix,
	const __global Real* vector,
	const __global int* columnIndeces,
	const __global int* nonzeroCount)
{
	// get element index
	const int i = get_global_id(0);

	// initialize result
	result[i] = 0;

	// for all non-zero row
	for(int j = 0; j < nonzeroCount[i]; j++)
	{
		// add matrix multiplied by vector
		result[i] += matrix[i * MAX_NONZERO_COUNT + j] * vector[columnIndeces[i * MAX_NONZERO_COUNT + j]];
	}
}