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
	const int count,
	__local Real* localValues)
{
	// get local size
	const int localSize = get_local_size(0);

	// get group index and size
	const int groupID = get_group_id(0);
	const int groupSize = get_num_groups(0);

	// get this element's index
	const int iLocal = get_local_id(0);
	const int iGlobal = (iLocal == 0) ? groupID : (groupSize + groupID*(localSize-1) + iLocal - 1);

	// copy values to local from grobal
	localValues[iLocal] = (iGlobal < count) ? values[iGlobal] : 0;

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// loop for reduction
	for(int nextOffset = 1; nextOffset < localSize; nextOffset *= 2)
	{
		// if this should work
		if(iLocal % (nextOffset*2) == 0)
		{
			// get next element's index
			const int jLocal = iLocal + nextOffset;

			// add next values to this
			localValues[iLocal] += localValues[jLocal];
		}

		// synchronize work items in this group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// only the first work-item in this group
	if(iLocal == 0)
	{
		// store sum of this group to global value
		values[groupID] = localValues[0];
	}
}

//! Sum array by reduction ver. 2
/*!
	\param values target array
	\param count number of elements
	\param nextOffset offset of next element of this element
*/
__kernel void ReductionSum2(
	__global Real* values,
	const int count,
	__local Real* localValues)
{
	// get local size
	const int localSize = get_local_size(0);

	// get group index and size
	const int groupID = get_group_id(0);
	const int groupSize = get_num_groups(0);

	// get this element's index
	const int iLocal = get_local_id(0);
	const int iGlobal = (iLocal == 0) ? groupID : (groupSize + groupID*(localSize-1) + iLocal - 1);

	// copy values from grobal to local
	localValues[iLocal] = (iGlobal < count) ? values[iGlobal] : 0;

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// for each half
	for(int halfSize = localSize/2; halfSize >= 1; halfSize /= 2)
	{
		// get second half element's index
		const int jLocal = iLocal + halfSize;

		// only in first half
		if(iLocal < halfSize)
		{
			// add second half values to this
			localValues[iLocal] += localValues[jLocal];
		}

		// synchronize work items in this group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// only the first work-item in this group
	if(iLocal == 0)
	{
		// store sum of this group to global value
		values[groupID] = localValues[0];
	}
}

//! Sum array by reduction ver. 3
/*!
	\param values target array
	\param count number of elements
	\param nextOffset offset of next element of this element
*/
__kernel void ReductionSum3(
	__global Real* values,
	const int count,
	__local Real* localValues)
{
	// get local size
	const int localSize = get_local_size(0);

	// get group index and size
	const int groupID = get_group_id(0);
	const int groupSize = get_num_groups(0);

	// get this element's index
	const int iLocal = get_local_id(0);
	const int iGlobal1 = 2*( (iLocal == 0) ? groupID : (groupSize + groupID*(localSize-1) + iLocal - 1));
	const int iGlobal2 = iGlobal1 + 1;

	// copy values from 2 global values to local
	localValues[iLocal] = 
	(iGlobal1 < count) ?
		values[iGlobal1] + ((iGlobal2 < count) ? values[iGlobal2] : 0)
		: 0;

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// for each half
	for(int halfSize = localSize/2; halfSize >= 1; halfSize /= 2)
	{
		// get second half element's index
		const int jLocal = iLocal + halfSize;

		// only in first half
		if(iLocal < halfSize)
		{
			// add second half values to this
			localValues[iLocal] += localValues[jLocal];
		}

		// synchronize work items in this group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// only the first work-item in this group
	if(iLocal == 0)
	{
		// store sum of this group to global value
		values[groupID] = localValues[0];
	}
}

//! Sum array by reduction ver. 4
/*!
	\param values target array
	\param count number of elements
	\param nextOffset offset of next element of this element
*/
__kernel void ReductionSum4(
	__global Real* values,
	const int count,
	__local Real* localValues)
{
	// get local size
	const int localSize = get_local_size(0);

	// get group index and size
	const int groupID = get_group_id(0);
	const int groupSize = get_num_groups(0);

	// get this element's index
	const int iLocal = get_local_id(0);
	const int iGlobal1 = ( (iLocal == 0) ? groupID : (groupSize + groupID*(localSize-1) + iLocal - 1) )<<1;
	const int iGlobal2 = iGlobal1 + 1;

	// copy values from 2 global values to local
	localValues[iLocal] = 
	(iGlobal1 < count) ?
		values[iGlobal1] + ((iGlobal2 < count) ? values[iGlobal2] : 0)
		: 0;

	// synchronize work items in this group
	barrier(CLK_LOCAL_MEM_FENCE);

	// for each half
	for(int halfSize = localSize>>1; halfSize >= 1; halfSize >>= 1)
	{
		// get second half element's index
		const int jLocal = iLocal + halfSize;

		// only in first half
		if(iLocal < halfSize)
		{
			// add second half values to this
			localValues[iLocal] += localValues[jLocal];
		}

		// synchronize work items in this group
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// only the first work-item in this group
	if(iLocal == 0)
	{
		// store sum of this group to global value
		values[groupID] = localValues[0];
	}
}