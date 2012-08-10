#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

//! REAL is provided by compiler option
typedef REAL Real;


//! Add each vector element
/*!
	\param answer vector which answer is stored to
	\param left adding vector
	\param right added vector
	\param C coefficient for added vector
*/
__kernel void AddEachVector(
	__global Real* answer,
	__global const Real* left,
	__global const Real* right)
{
	// get element index
	int i = get_global_id(0);

	// add each element
	answer[i] = left[i] + right[i];
}