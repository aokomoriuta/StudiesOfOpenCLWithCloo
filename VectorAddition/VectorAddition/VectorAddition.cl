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


//! Add one element as vector per one work-item
/*!
	\param result vector which result is stored to
	\param left adding vector
	\param right added vector
	\param C coefficient for added vector
*/
__kernel void AddOneVector(
	__global Real* result,
	const __global const Real* left,
	const __global const Real* right)
{
	// get element index
	int i = get_global_id(0);

	// load as vector
	RealV leftVector =  VLOADN(i, left);
	RealV rightVector = VLOADN(i, right);

	// add as vector
	RealV resultVector = leftVector + rightVector;

	// store result
	VSTOREN(resultVector, i, result);
}


//! Add more than one element per one work-item
/*!
	\param result vector which result is stored to
	\param left adding vector
	\param right added vector
	\param C coefficient for added vector
*/
__kernel void AddMoreElement(
	__global Real* result,
	const __global const Real* left,
	const __global const Real* right)
{
	// get element index
	int i = get_global_id(0);

	//// for all target element by this work-item
	//for(int j = 0; j < COUNT_PER_WORKITEM; j++)
	//{
	//	// add element
	//	result[i+j] = left[i+j] + right[i+j];
	//}
#define ADDONE(j) result[COUNT_PER_WORKITEM*i+j]=left[COUNT_PER_WORKITEM*i+j]+right[COUNT_PER_WORKITEM*i+j];
	ADDONE( 0)
	ADDONE( 1)
	ADDONE( 2)
	ADDONE( 3)
	ADDONE( 4)
	ADDONE( 5)
	ADDONE( 6)
	ADDONE( 7)
	ADDONE( 8)
	ADDONE( 9)
	ADDONE(10)
	ADDONE(11)
	ADDONE(12)
	ADDONE(13)
	ADDONE(14)
	ADDONE(15)
}