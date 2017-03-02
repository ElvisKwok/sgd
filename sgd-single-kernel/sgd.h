#ifndef __GSGD_H_
#define __GSGD_H_

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

#include <cuda_runtime.h>
#include <device_launch_parameters.h>	//  vs����δ�������ʾ
#include <device_functions.h>

#include <iostream>
#include <algorithm>

#include "gpusgd_serial.h"


using namespace std;

void solveByGPU(int *a, int *b, int *c, int size);

#endif