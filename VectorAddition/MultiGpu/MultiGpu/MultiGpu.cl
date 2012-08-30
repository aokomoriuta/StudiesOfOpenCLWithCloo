#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

//! REAL is provided by compiler option
typedef REAL Real;
typedef REALV RealV;

//! Add one element per one work-item
/*!
	\param result vector which result is stored to
	\param left adding vector
	\param right added vector
	\param C coefficient for added vector
*/
__kernel void AddOneElement(
	__global Real* result,
	const __global const Real* left,
	const __global const Real* right)
{
	// get element index
	int i = get_global_id(0);

	// add each element
	result[i] = left[i] + right[i];
}