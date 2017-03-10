#include "sgd.h"
#define CHECK(res) if(res!=cudaSuccess){exit(-1);} 


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
								 typeRate *matrixUser,
								 typeRate *matrixItem,
								 int K						// 隐含向量维数
								)
{
	typeRate predictRate = 0;
	for (int k = 0; k < K; ++k)
	{
		predictRate += (*(matrixUser + userIdx*K +k)) * (*(matrixItem + itemIdx*K+k));
	}
	return predictRate;
}


// sgd一次更新操作
__device__ void sgdUpdate(
						  typeRate rate,
						  int userIdx,
						  int itemIdx,
						  typeRate *matrixUser,
						  typeRate *matrixItem,
						  int K,					// 隐含向量维数
						  double lambda,			// 正则化系数
						  double gamma				// 学习率
						 )
{
	typeRate predictRate = innerProduct(userIdx, itemIdx, matrixUser, matrixItem, K);
	typeRate error = rate - predictRate;
	typeRate tmp;
	for (int k = 0; k < K; ++k)
	{
		/*
		(*(matrixUser + userIdx*K + k)) += (gamma * (2 * error * (*(matrixItem + itemIdx*K + k)) - lambda * (*(matrixUser + userIdx*K + k))));
		(*(matrixItem + itemIdx*K + k)) += (gamma * (2 * error * (*(matrixUser + userIdx*K + k)) - lambda * (*(matrixItem + itemIdx*K + k))));
		*/
		tmp = (*(matrixUser + userIdx*K + k));
		(*(matrixUser + userIdx*K + k)) += (gamma * (error * (*(matrixItem + itemIdx*K + k)) - lambda * tmp));
		(*(matrixItem + itemIdx*K + k)) += (gamma * (error * tmp - lambda * (*(matrixItem + itemIdx*K + k))));
	}
}


__global__ void sgd_kernel(
							sRateNode *d_rateNodeArray,
							typeRate *d_matrixUser,
							typeRate *d_matrixItem,
							sWorkset *d_worksetArray,
							sWorkseg *d_mWorkseg,
							int *d_matrixPattern,
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
	int bid = *(d_matrixPattern + s*subBlockNumL + bidx);
	
	// c++风格 end为下一个位置
	if ((bid == -1) || d_worksetArray[bid].beg == d_worksetArray[bid].end)	// 空块的pattern默认初始化为-1
	{
		return;
	}

	for (int tag = 0; tag < subBlockLen; ++tag)
	{
		int from = (*(d_mWorkseg + bid*subBlockLen + tag)).from;
		int to = (*(d_mWorkseg + bid*subBlockLen + tag)).to;
		// QUESTION: tid从0开始？
		// c++风格 end为下一个位置
		for (int iRate = from + tid; iRate < to; iRate += blockDim.x)	// iRate为 属于子块bid && 标签为tag 的评价值 在rateNodeArray数组的下标
		{
			typeRate rate = d_rateNodeArray[iRate].rate;
			int userIdx = d_rateNodeArray[iRate].u - 1;
			int itemIdx = d_rateNodeArray[iRate].i - 1;
			sgdUpdate(rate, userIdx, itemIdx, d_matrixUser, d_matrixItem, K, lambda, gamma);
			printf("userIdx = %d, itemIdx = %d\n", userIdx, itemIdx);
		}

		// wait for all threads in this block to arrive here(i.e. current tag finish)
		__syncthreads();
	}

}


typeRate computeRMSE(sRateNode *rateNodeArray, typeRate *matrixUser,typeRate *matrixItem, int NNZ)
{
	int userIdx;
	int itemIdx;
	typeRate predictRate = 0.0;
	typeRate err_sum = 0.0;
	for (int i = 0; i < NNZ; ++i)
	{
		userIdx = rateNodeArray[i].u - 1;
		itemIdx = rateNodeArray[i].i - 1;
		predictRate = innerProduct(matrixUser, matrixItem, userIdx, itemIdx);
		//cout << "(" << userIdx+1 << ", " << itemIdx+1 << "): "<<predictRate << endl;
		err_sum += pow((rateNodeArray[i].rate - predictRate), 2);
	}
	return sqrt(err_sum / NNZ);
}

void solveByGPU(
				sRateNode *rateNodeArray,
				typeRate *matrixUser,
				typeRate *matrixItem,
				sWorkset *worksetArray,
				sWorkseg *mWorkseg,
				int *matrixPattern,
				int subBlockNumL,			// subBlockNumL * subBlockNumL个子块 
				int subBlockLen,			// 子块大小为 subBlockLen * subBlockLen
				double lambda,				// 正则化系数
				double gamma,				// 学习率
				int NNZ						// 评价值个数
			   )
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	// QUESTION: dim_grid.x(block维度)、dim_block.x(thread维度)设置？
	dim3 dim_grid, dim_block;
	dim_block.x = min(prop.maxThreadsDim[0], prop.maxThreadsPerBlock);
	/*
	if (dim_block.x >= size) {	
		dim_block.x = size;
	}
	dim_grid.x = size / dim_block.x;
	if (size % dim_block.x != 0) {
		dim_grid.x++;
	}
	*/
	dim_grid.x = subBlockNumL;


	sRateNode *d_rateNodeArray;
	typeRate *d_matrixUser;
	typeRate *d_matrixItem;
	sWorkset *d_worksetArray;
	sWorkseg *d_mWorkseg;
	int *d_matrixPattern;
	
	int subBlockNum = subBlockNumL*subBlockNumL;

	cudaError_t res;
	res = cudaMalloc((void**)(&d_rateNodeArray), NNZ * sizeof(sRateNode)); CHECK(res)
	res = cudaMalloc((void**)(&d_matrixUser), M*K * sizeof(typeRate)); CHECK(res)
	res = cudaMalloc((void**)(&d_matrixItem), N*K * sizeof(typeRate)); CHECK(res)
	res = cudaMalloc((void**)(&d_worksetArray), subBlockNum * sizeof(sWorkset)); CHECK(res)
	res = cudaMalloc((void**)(&d_mWorkseg), subBlockNum*subBlockLen * sizeof(sWorkseg)); CHECK(res)
	res = cudaMalloc((void**)(&d_matrixPattern), subBlockNum * sizeof(int)); CHECK(res)


	res = cudaMemcpy(d_rateNodeArray, rateNodeArray, NNZ * sizeof(sRateNode), cudaMemcpyHostToDevice); CHECK(res)
	res = cudaMemcpy(d_matrixUser, matrixUser, M*K * sizeof(typeRate), cudaMemcpyHostToDevice); CHECK(res)
	res = cudaMemcpy(d_matrixItem, matrixItem, N*K * sizeof(typeRate), cudaMemcpyHostToDevice); CHECK(res)
	res = cudaMemcpy(d_worksetArray, worksetArray, subBlockNum * sizeof(sWorkset), cudaMemcpyHostToDevice); CHECK(res)
	res = cudaMemcpy(d_mWorkseg, mWorkseg, subBlockNum*subBlockLen * sizeof(sWorkseg), cudaMemcpyHostToDevice); CHECK(res)
	res = cudaMemcpy(d_matrixPattern, matrixPattern, subBlockNum * sizeof(int), cudaMemcpyHostToDevice); CHECK(res)

		/*
		sgd_kernel << < dim_grid, dim_block >> >(dev_c, dev_a, dev_b, size);
		*/
		for (int iter = 0; iter < MAX_ITER; ++iter)
		{
			for (int s = 0; s < subBlockNumL; ++s)
			{
				sgd_kernel <<< dim_grid, dim_block >>> (
															d_rateNodeArray,
															d_matrixUser,
															d_matrixItem,
															d_worksetArray,
															d_mWorkseg,
															d_matrixPattern,
															s,						// 第s个模式
															subBlockNumL,			// subBlockNumL * subBlockNumL个子块 
															subBlockLen,			// 子块大小为 subBlockLen * subBlockLen
															K,						// 隐含向量维数
															lambda,					// 正则化系数
															gamma					// 学习率
														);
			}
			//cudaMemcpy(matrixUser, d_matrixUser, M*K * sizeof(typeRate), cudaMemcpyDeviceToHost); CHECK(res)
			//printMatrix(matrixUser, M, K);
		}
	
	res = cudaMemcpy(matrixUser, d_matrixUser, M*K * sizeof(typeRate), cudaMemcpyDeviceToHost); CHECK(res)
	res = cudaMemcpy(matrixItem, d_matrixItem, N*K * sizeof(typeRate), cudaMemcpyDeviceToHost); CHECK(res)

	cout << "RMSE: " << computeRMSE(rateNodeArray, matrixUser, matrixItem, NNZ) << endl;

	cudaFree(d_rateNodeArray);
	cudaFree(d_matrixUser);
	cudaFree(d_matrixItem);
	cudaFree(d_worksetArray);
	cudaFree(d_mWorkseg);
	cudaFree(d_matrixPattern);
}