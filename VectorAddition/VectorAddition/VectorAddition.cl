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

	// unrolled loop to add element
	#define ADD_ONE(j) result[COUNT_PER_WORKITEM*i+j]=left[COUNT_PER_WORKITEM*i+j]+right[COUNT_PER_WORKITEM*i+j];
		ADD_ONE( 0)
		ADD_ONE( 1)
		ADD_ONE( 2)
		ADD_ONE( 3)
		ADD_ONE( 4)
		ADD_ONE( 5)
		ADD_ONE( 6)
		ADD_ONE( 7)
		ADD_ONE( 8)
		ADD_ONE( 9)
		ADD_ONE(10)
		ADD_ONE(11)
		ADD_ONE(12)
		ADD_ONE(13)
		ADD_ONE(14)
		ADD_ONE(15)
	#undef ADD_ONE
}

//! Add more than one element per one work-item
/*!
	\param result vector which result is stored to
	\param left adding vector
	\param right added vector
	\param C coefficient for added vector
*/
__kernel void AddMoreVector(
	__global Real* result,
	const __global const Real* left,
	const __global const Real* right)
{
	// get element index
	int i = get_global_id(0);

	// load left vector
	#define LEFT(j) VLOADN(COUNT_PER_WORKITEM*i+j,left)

	// load right vector
	#define RIGHT(j) VLOADN(COUNT_PER_WORKITEM*i+j,right)

	// get left+right
	#define ADD(j) LEFT(j)+RIGHT(j)

	// store result
	#define STORE(j) VSTOREN(ADD(j),COUNT_PER_WORKITEM*i+j,result);
	
	// unrolled loop to add vector
		STORE( 0)
		STORE( 1)
		STORE( 2)
		STORE( 3)
		STORE( 4)
		STORE( 5)
		STORE( 6)
		STORE( 7)
		STORE( 8)
		STORE( 9)
		STORE(10)
		STORE(11)
		STORE(12)
		STORE(13)
		STORE(14)
		STORE(15)
#undef LEFT
#undef RIGHT
#undef ADD
#undef STORE
}