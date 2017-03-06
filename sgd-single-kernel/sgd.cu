#include "sgd.h"

// unused code
/*
__global__ void addKernel_3(int *c, const int *a, const int *b, unsigned int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadIdx.x： 某个block内的线程下标
	int stride = blockDim.x * gridDim.x;	// blockDim.x：1个block有多少个thread, gridDim.x：1个grid有多少个block
	for (int i = tid; i < size; i += stride)
	{
		c[tid] = a[tid] + b[tid];
	}
}


void solveByGPU(int *a, int *b, int *c, int size)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	dim3 dim_grid, dim_block;
	dim_block.x = min(prop.maxThreadsDim[0], prop.maxThreadsPerBlock);
	if (dim_block.x >= size) {
		dim_block.x = size;
	}
	dim_grid.x = size / dim_block.x;
	if (size % dim_block.x != 0) {
		dim_grid.x++;
	}


	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaMalloc((void**)&dev_a, size * sizeof(int));
	cudaMalloc((void**)&dev_b, size * sizeof(int));
	cudaMalloc((void**)&dev_c, size * sizeof(int));
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	addKernel_3 <<< dim_grid, dim_block >> >(dev_c, dev_a, dev_b, size);

	cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

*/


// TO-DO:
// 向量内积
__device__ typeRate innerProduct(
								 int userIdx,
								 int itemIdx,
								 typeRate **matrixUser,
								 typeRate **matrixItem,
								 int K						// 隐含向量维数
								)
{
	typeRate predictRate = 0;
	for (int k = 0; k < K; ++k)
	{
		predictRate += matrixUser[userIdx][k] * matrixItem[itemIdx][k];
	}
	return predictRate;
}


// sgd一次更新操作
__device__ void sgdUpdate(
						  typeRate rate,
						  int userIdx,
						  int itemIdx,
						  typeRate **matrixUser,
						  typeRate **matrixItem,
						  int K,					// 隐含向量维数
						  double lambda,			// 正则化系数
						  double gamma				// 学习率
						 )
{
	typeRate predictRate = innerProduct(userIdx, itemIdx, matrixUser, matrixItem, K);
	typeRate error = rate - predictRate;
	for (int k = 0; k < K; ++k)
	{
		matrixUser[userIdx][k] += (gamma * (2 * error * matrixItem[itemIdx][k] - lambda * matrixUser[userIdx][k]));
		matrixItem[itemIdx][k] += (gamma * (2 * error * matrixUser[userIdx][k] - lambda * matrixItem[itemIdx][k]));
	}
}


__global__ void sgd_kernel(
							sRateNode *rateNodeArray,
							typeRate **d_matrixUser,
							typeRate **d_matrixItem,
							sWorkset *d_worksetArray,
							sWorkseg **d_mWorkseg,
							int **d_matrixPattern,
							int s,						// 第s个模式
							int subBlockNumL,			// subBlockNumL * subBlockNumL个子块 
							int subBlockLen,			// 子块大小为 subBlockLen * subBlockLen
							int K,						// 隐含向量维数
							double lambda,				// 正则化系数
							double gamma				// 学习率
						   )
{
	int bidx = blockIdx.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// TO-DO: 判断bidx越界？ subBlockNumL
	if (bidx > subBlockNumL)
	{
		return;
	}
	int bid = d_matrixPattern[s][bidx];
	
	if (d_worksetArray[bid].beg == d_worksetArray[bid].end)
	{
		return;
	}

	for (int tag = 0; tag < subBlockLen; ++tag)
	{
		int from = d_mWorkseg[bid][tag].from;
		int to = d_mWorkseg[bid][tag].to;
		for (int iRate = from + tid; iRate < to; iRate += blockDim.x)	// iRate为 属于子块bid && 标签为tag 的评价值 在rateNodeArray数组的下标
		{
			typeRate rate = rateNodeArray[iRate].rate;
			int userIdx = rateNodeArray[iRate].u;
			int itemIdx = rateNodeArray[iRate].i;
			sgdUpdate(rate, userIdx, itemIdx, d_matrixUser, d_matrixItem, K, lambda, gamma);
		}

		// wait for all threads in this block to arrive here(i.e. current tag finish)
		__syncthreads();
	}

}


void solveByGPU(int *a, int *b, int *c, int size)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	dim3 dim_grid, dim_block;
	dim_block.x = min(prop.maxThreadsDim[0], prop.maxThreadsPerBlock);
	if (dim_block.x >= size) {
		dim_block.x = size;
	}
	dim_grid.x = size / dim_block.x;
	if (size % dim_block.x != 0) {
		dim_grid.x++;
	}


	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaMalloc((void**)&dev_a, size * sizeof(int));
	cudaMalloc((void**)&dev_b, size * sizeof(int));
	cudaMalloc((void**)&dev_c, size * sizeof(int));
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	/*
	sgd_kernel << < dim_grid, dim_block >> >(dev_c, dev_a, dev_b, size);
	*/

	cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}