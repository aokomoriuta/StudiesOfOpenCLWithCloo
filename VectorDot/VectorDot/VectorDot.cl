#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

//! REAL is provided by compiler option
typedef REAL Real;

//! Multyply one element per one work-item
/*!
	\param result vector which result is stored to
	\param left multiplied vector
	\param right multiplying vector
*/
__kernel void MultyplyEachElement(
	__global Real* result,
	const __global const Real* left,
	const __global const Real* right)
{
	// get element index
	const int i = get_global_id(0);

	// multiply each element
	result[i] = left[i] * right[i];
}

//! Sum array by reduction ver. 0
/*!
	\param values target array
	\param count number of elements
	\param nextOffset offset of next element of this element
*/
__kernel void ReductionSum0(
	__global Real* values,
	const int count,
	const int nextOffset)
{
	// get element index
	const int i = get_global_id(0) * nextOffset * 2;
	const int j = i + nextOffset;

	// only in region of target
	if(j < count)
	{
		// add next values to this
		values[i] += values[j];
	}
}

//! Sum array by reduction ver. 1
/*!
	\param values target array
	\param count number of elements
	\param nextOffset offset of next element of this element
*/
__kernel void ReductionSum1(
	__global Real* values,
	const int count)
{
	// get element index
	const int i = get_global_id(0);
	const int j = i + get_global_size(0);

	// only in region of target
	if(j < count)
	{
		// add next values to this
		values[i] += values[j];
	}
}