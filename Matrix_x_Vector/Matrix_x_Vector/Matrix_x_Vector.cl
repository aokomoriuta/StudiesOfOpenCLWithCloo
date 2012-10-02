#if __OPENCL__VERSION__ <= __CL_VERSION_1_1
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

//! REAL is provided by compiler option
typedef REAL Real;

//! Multiply matrix by vector ver.0
/*!
	\param result vector which result is stored to
	\param left multiplied vector
	\param right multiplying vector
*/
__kernel void Matrix_x_Vector0(
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

//! Multiply matrix by vector ver.1
/*!
	\param result vector which result is stored to
	\param left multiplied vector
	\param right multiplying vector
*/
__kernel void Matrix_x_Vector1(
	__global Real* result,
	const __global Real* matrix,
	const __global Real* vector,
	const __global int* columnIndeces,
	const __global int* nonzeroCount)
{
	// get element index
	const int i = get_global_id(0);

	// initialize result
	Real thisResult = 0;

	// for all non-zero row
	for(int j = 0; j < nonzeroCount[i]; j++)
	{
		// add matrix multiplied by vector
		thisResult += matrix[i * MAX_NONZERO_COUNT + j] * vector[columnIndeces[i * MAX_NONZERO_COUNT + j]];
	}

	// store result
	result[i] = thisResult;
}

//! Multiply matrix by vector ver.2
/*!
	\param result vector which result is stored to
	\param left multiplied vector
	\param right multiplying vector
*/
__kernel void Matrix_x_Vector2(
	const int count,
	__global Real* result,
	const __global Real* matrix,
	const __global Real* vector,
	const __global int* columnIndeces,
	const __global int* nonzeroCount,
	const int bufferSize,
	__local Real* localVector)
{
	// get element index
	const int globalID = get_global_id(0);

	// get other index and size
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int groupID = get_group_id(0);
	const int vectorFirstID = max(0, localSize *  groupID - bufferSize);
	const int vectorLastID  = min(count, localSize * (groupID + 1) + bufferSize); 

	// for each local vector
	for(int i = localID; i < vectorLastID  - vectorFirstID; i += localSize)
	{
		// copy from global
		localVector[i] = vector[vectorFirstID + i];
	}

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// ignore if index is larger than row count
	if(globalID > count) return;

	// initialize result
	Real thisResult = 0;

	// for all non-zero row
	for(int j = 0; j < nonzeroCount[globalID]; j++)
	{
		// add matrix multiplied by vector
		thisResult += matrix[globalID * MAX_NONZERO_COUNT + j] * localVector[columnIndeces[globalID * MAX_NONZERO_COUNT + j] - vectorFirstID];
	}

	// store result
	result[globalID] = thisResult;
}