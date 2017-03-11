#ifndef __GSGD_H_
#define __GSGD_H_

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

#include <cuda_runtime.h>
#include <device_launch_parameters.h>	//  vs����δ�������ʾ
#include <device_functions.h>
#include <cublas_v2.h>

#include <iostream>
#include <algorithm>

#include "gpusgd_serial.h"
#include "parameter.h"

using namespace std;

void solveByGPU(
	sRateNode *rateNodeArray,
	typeRate *matrixUser,
	typeRate *matrixItem,
	sWorkset *worksetArray,
	sWorkseg *mWorkseg,
	int *matrixPattern,
	int subBlockNumL,			// subBlockNumL * subBlockNumL���ӿ� 
	int subBlockLen,			// �ӿ��СΪ subBlockLen * subBlockLen
	double lambda,				// ����ϵ��
	double gamma,				// ѧϰ��
	int NNZ						// ����ֵ����
	);

#endif