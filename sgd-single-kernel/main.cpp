#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <string>
#include <vector>

#include "gpusgd_serial.h"
#include "basic_func.h"

using namespace std;

ofstream outfile("result/output.txt");

#define TEST_STEPS 80000
//#define TEST_STEPS 650000
#define STEP_PRINT 160

#define ROW_NUM 5
#define COLUMN_NUM 4
#define K_NUM 2

double dMatrixP[ROW_NUM][K_NUM];
double dMatrixQ[K_NUM][COLUMN_NUM];
double dMatrixR[ROW_NUM][COLUMN_NUM] = { { 5, 3, 0, 1 }, { 4, 0, 0, 1 }, { 1, 1, 0, 5 }, { 1, 0, 0, 4 }, { 0, 1, 5, 4 } };
double dMatrixRNext[ROW_NUM][COLUMN_NUM];

void printMatrix(double dMatrix[ROW_NUM][COLUMN_NUM])
{
	for (int i = 0; i < ROW_NUM; ++i)
	{
		for (int j = 0; j < COLUMN_NUM; ++j)
		{
			outfile << dMatrix[i][j] << "\t";
		}
		outfile << endl;
	}
}

// 计算向量内积 dot(P[i, :], Q[:, j])
double dot(double dMatrixP[ROW_NUM][K_NUM], double dMatrixQ[K_NUM][COLUMN_NUM], const int iRow, const int iColumn, const int iIndex)
{
	double dInnerProduct = 0;
	for (int i = 0; i < iIndex; ++i)
	{
		dInnerProduct += (dMatrixP[iRow][i] * dMatrixQ[i][iColumn]);
	}
	return dInnerProduct;
}

// 矩阵相乘
void matrixMultiply(double dMatrixResult[ROW_NUM][COLUMN_NUM], double dMatrixP[ROW_NUM][K_NUM], double dMatrixQ[K_NUM][COLUMN_NUM], const int M, const int N, const int K)
{

	for (int k = 0; k < K; ++k)
	{
		for (int i = 0; i < M; ++i)
		{
			double dTmp = dMatrixP[i][k];
			for (int j = 0; j < N; ++j)
			{
				dMatrixResult[i][j] += (dTmp * dMatrixQ[k][j]);
			}
		}
	}
}

// 矩阵相加
void matrixAdd(double dMatrixA[ROW_NUM][K_NUM], double dMatrixB[ROW_NUM][K_NUM], double dMatrixC[ROW_NUM][K_NUM])
{
	for (int i = 0; i < ROW_NUM; ++i)
	{
		for (int j = 0; j < K_NUM; ++j)
		{
			dMatrixC[i][j] = dMatrixA[i][j] + dMatrixB[i][j];
		}
	}
}


void sgd(double dMatrixR[ROW_NUM][COLUMN_NUM], double dMatrixP[ROW_NUM][K_NUM], double dMatrixQ[K_NUM][COLUMN_NUM], 
		 const int iRowN, const int iColumnN, const int iFeatureN, const int iStepN = 650000, const double dAlpha = 0.0002, const double dBeta = 0.02)
{
	for (int iStep = 0; iStep < iStepN; ++iStep)
	{
		for (int iRow = 0; iRow < iRowN; ++iRow)
		{
			for (int iColumn = 0; iColumn < iColumnN; ++iColumn)
			{
				if (dMatrixR[iRow][iColumn] > 0)
				{
					// 计算向量内积 dot(P[i,:], Q[:,j])
					double dInnerProduct = dot(dMatrixP, dMatrixQ, iRow, iColumn, K_NUM);

					double dErrorIJ = dMatrixR[iRow][iColumn] - dInnerProduct;
					for (int iFeature = 0; iFeature < iFeatureN; ++iFeature)
					{
						dMatrixP[iRow][iFeature] += (dAlpha * (2 * dErrorIJ * dMatrixQ[iFeature][iColumn] - dBeta * dMatrixP[iRow][iFeature]));
						dMatrixQ[iFeature][iColumn] += (dAlpha * (2 * dErrorIJ * dMatrixP[iRow][iFeature] - dBeta * dMatrixQ[iFeature][iColumn]));
					}
				}
			}
		}

		//matrixMultiply(dMatrixRNext, dMatrixP, dMatrixQ, ROW_NUM, COLUMN_NUM, K_NUM);
		double dErrorSum = 0;
		for (int iRow = 0; iRow < iRowN; ++iRow)
		{
			for (int iColumn = 0; iColumn < iColumnN; ++iColumn)
			{
				if (dMatrixR[iRow][iColumn] > 0)
				{
					double dInnerProduct = dot(dMatrixP, dMatrixQ, iRow, iColumn, K_NUM);
					dErrorSum += pow((dMatrixR[iRow][iColumn] - dInnerProduct), 2);
					/*for (int iFeature = 0; iFeature < iFeatureN; ++iFeature)
					{
						dErrorSum += ((dBeta / 2) * (pow(dMatrixP[iRow][iFeature], 2) + pow(dMatrixQ[iFeature][iColumn], 2)));
					}*/
				}
			}
		}
		if (iStep % STEP_PRINT == 0) {
			outfile << "step: " << iStep << "\tdErrorSum: " << dErrorSum << endl;
		}
		if (dErrorSum < 0.001)
		{
			break;
		}
	}
}



/*
// wrong
typedef struct ImplicitData
{
	double y[K_NUM];
} ImplicitData;
ImplicitData sImplicitData[ROW_NUM][COLUMN_NUM];
double dUserRateCount[ROW_NUM];

void initImplicitData(double dMatrixR[ROW_NUM][COLUMN_NUM])
{
	double dUserRateSum[ROW_NUM];
	for (int iRow = 0; iRow < ROW_NUM; ++iRow)
	{
		dUserRateSum[iRow] = 0;
		dUserRateCount[iRow] = 0;
		for (int iColumn = 0; iColumn < COLUMN_NUM; ++iColumn)
		{
			if (dMatrixR[iRow][iColumn] > 0)
			{
				for (int k = 0; k < K_NUM; ++k)
				{
					sImplicitData[iRow][iColumn].y[k] = 1;
				}
				dUserRateSum[iRow] += dMatrixR[iRow][iColumn];
				dUserRateCount[iRow]++;
			}
			else
			{
				//sImplicitData[iRow][iColumn].y = 0;
			}
		}
	}
}

void sgdPlusPlus(double dMatrixR[ROW_NUM][COLUMN_NUM], double dMatrixP[ROW_NUM][K_NUM], double dMatrixQ[K_NUM][COLUMN_NUM],
	const int iRowN, const int iColumnN, const int iFeatureN, const int iStepN = 650000, const double dAlpha = 0.0002, const double dBeta = 0.02)
{
	initImplicitData(dMatrixR);
	for (int iStep = 0; iStep < iStepN; ++iStep)
	{
		for (int iRow = 0; iRow < iRowN; ++iRow)
		{
			for (int iColumn = 0; iColumn < iColumnN; ++iColumn)
			{
				if (dMatrixR[iRow][iColumn] > 0)
				{
					// 计算向量内积 dot(P[i,:], Q[:,j])
					double dInnerProduct = dot(dMatrixP, dMatrixQ, iRow, iColumn, K_NUM);

					double dErrorIJ = dMatrixR[iRow][iColumn] - dInnerProduct;
					//double dImplicitPart = sImplicitData[iRow][iColumn].y / sqrt(dUserRateCount[iRow]);
					double dImplicitPart[K_NUM];
					for (int k = 0; k < K_NUM; ++k)
					{
						dImplicitPart[k] = sImplicitData[iRow][iColumn].y[k] / sqrt(dUserRateCount[iRow]);
					}
					for (int iFeature = 0; iFeature < iFeatureN; ++iFeature)
					{
						dMatrixP[iRow][iFeature] += (dAlpha * (2 * dErrorIJ * dMatrixQ[iFeature][iColumn] - dBeta * dMatrixP[iRow][iFeature]));
						dMatrixQ[iFeature][iColumn] += (dAlpha * (2 * dErrorIJ * (dMatrixP[iRow][iFeature] + dImplicitPart[iFeature]) - dBeta * dMatrixQ[iFeature][iColumn]));
						sImplicitData[iRow][iColumn].y[iFeature] += (dAlpha * (2 * dErrorIJ * dMatrixQ[iFeature][iColumn] / sqrt(dUserRateCount[iRow]) - dBeta * sImplicitData[iRow][iColumn].y[iFeature]));
					}
				}
			}
		}

		//matrixMultiply(dMatrixRNext, dMatrixP, dMatrixQ, ROW_NUM, COLUMN_NUM, K_NUM);
		double dErrorSum = 0;
		for (int iRow = 0; iRow < iRowN; ++iRow)
		{
			for (int iColumn = 0; iColumn < iColumnN; ++iColumn)
			{
				if (dMatrixR[iRow][iColumn] > 0)
				{
					double dInnerProduct = dot(dMatrixP, dMatrixQ, iRow, iColumn, K_NUM); // to-do: dMatrixQ + implicitData
					dErrorSum += pow((dMatrixR[iRow][iColumn] - dInnerProduct), 2);
					for (int iFeature = 0; iFeature < iFeatureN; ++iFeature)
					{
						dErrorSum += ((dBeta / 2) * (pow(dMatrixP[iRow][iFeature], 2) + pow(dMatrixQ[iFeature][iColumn], 2)));
					}
				}
			}
		}
		if (iStep % 50000 == 0) {
			outfile << "step: " << iStep << "\tdErrorSum: " << dErrorSum << endl;
		}
		if (dErrorSum < 0.001)
		{
			break;
		}
	}
}
*/


/*
* new
*/

///*
// the inverse square root of the number of movie ratings
double *norm = new double[ROW_NUM];
// stores the y-values of each y[movie][feature]
//double **y = new double*[COLUMN_NUM];
double y[K_NUM][COLUMN_NUM]; // 后续可考虑转置表示，按行访问效率更高
// stores the sum of the y's: sumY[user][feature]
//double **sumY = new double*[ROW_NUM];
double sumY[ROW_NUM][K_NUM];

void computeNorms() {
	for (int iRow = 0; iRow < ROW_NUM; ++iRow)
	{
		int iUserRateCnt = 0;
		for (int iColumn = 0; iColumn < COLUMN_NUM; ++iColumn)
		{
			if (dMatrixR[iRow][iColumn] > 0)
			{
				iUserRateCnt++;
			}
		}
		norm[iRow] = pow(iUserRateCnt, -0.5);
	}
}
void initSumY()
{
	for (int iRow = 0; iRow < ROW_NUM; ++iRow)
	{
		for (int iColumn = 0; iColumn < COLUMN_NUM; ++iColumn)
		{
			if (dMatrixR[iRow][iColumn] > 0)
			{
				for (int j = 0; j < K_NUM; ++j)
				{
					sumY[iRow][j] += y[j][iColumn];
				}
			}
		}
		
	}
}

double predictRui(double dMatrixP[ROW_NUM][K_NUM], double dMatrixQ[K_NUM][COLUMN_NUM], double y[ROW_NUM][K_NUM], const int iRow, const int iColumn)
{
	double dMatrixBias[ROW_NUM][K_NUM];
	matrixAdd(dMatrixP, sumY, dMatrixBias);
	double dInnerProduct = dot(dMatrixBias, dMatrixQ, iRow, iColumn, K_NUM);
	return dInnerProduct;
}

void sgdPlusPlus(double dMatrixR[ROW_NUM][COLUMN_NUM], double dMatrixP[ROW_NUM][K_NUM], double dMatrixQ[K_NUM][COLUMN_NUM],
				 const int iRowN, const int iColumnN, const int iFeatureN, const int iStepN = 650000, const double dAlpha = 0.0002, const double dBeta = 0.02)
{
	computeNorms();
	initSumY();
	for (int iStep = 0; iStep < iStepN; ++iStep)
	{
		for (int iRow = 0; iRow < iRowN; ++iRow)
		{
			for (int iColumn = 0; iColumn < iColumnN; ++iColumn)
			{
				if (dMatrixR[iRow][iColumn] > 0)
				{
					// 计算向量内积 dot(P[i,:], Q[:,j])
					//double dInnerProduct = dot(dMatrixP, dMatrixQ, iRow, iColumn, K_NUM);
					double dInnerProduct = predictRui(dMatrixP, dMatrixQ, sumY, iRow, iColumn);
					double dErrorIJ = dMatrixR[iRow][iColumn] - dInnerProduct;
					
					for (int iFeature = 0; iFeature < iFeatureN; ++iFeature)
					{
						/*
						dMatrixP[iRow][iFeature] += (dAlpha * (2 * dErrorIJ * dMatrixQ[iFeature][iColumn] - dBeta * dMatrixP[iRow][iFeature]));
						dMatrixQ[iFeature][iColumn] += (dAlpha * (2 * dErrorIJ * (dMatrixP[iRow][iFeature] + norm[iRow] * sumY[iRow][iFeature]) - dBeta * dMatrixQ[iFeature][iColumn]));
						double yUpdate = (dAlpha * (2 * dErrorIJ * norm[iRow] * dMatrixQ[iFeature][iColumn] - dBeta * y[iFeature][iColumn]));
						y[iFeature][iColumn] += yUpdate;
						sumY[iRow][iFeature] += yUpdate;
						*/
						double oldUserFeature = dMatrixP[iRow][iFeature];
						double oldItemFeature = dMatrixQ[iFeature][iColumn];
						dMatrixP[iRow][iFeature] += (dAlpha * (2 * dErrorIJ * oldItemFeature - dBeta * oldUserFeature));
						dMatrixQ[iFeature][iColumn] += (dAlpha * (2 * dErrorIJ * (oldUserFeature + norm[iRow] * sumY[iRow][iFeature]) - dBeta * oldItemFeature));

						for (int iColumn2 = 0; iColumn2 < iColumnN; ++iColumn2)
						{
							if (dMatrixR[iRow][iColumn2] > 0)
							{
								double yUpdate = (dAlpha * (2 * dErrorIJ * norm[iRow] * oldItemFeature - dBeta * y[iFeature][iColumn2]));
								y[iFeature][iColumn2] += yUpdate;
								sumY[iRow][iFeature] += yUpdate;
							}
						}
					}
				}
			}
		}

		double dErrorSum = 0; 

		for (int iRow = 0; iRow < iRowN; ++iRow)
		{
			for (int iColumn = 0; iColumn < iColumnN; ++iColumn)
			{
				if (dMatrixR[iRow][iColumn] > 0)
				{
					//double dInnerProduct = dot(dMatrixP, dMatrixQ, iRow, iColumn, K_NUM); // to-do: dMatrixQ + implicitData
					double dInnerProduct = predictRui(dMatrixP, dMatrixQ, sumY, iRow, iColumn);
					dErrorSum += pow((dMatrixR[iRow][iColumn] - dInnerProduct), 2);
					/*
					for (int iFeature = 0; iFeature < iFeatureN; ++iFeature)
					{
						dErrorSum += ((dBeta / 2) * (pow(dMatrixP[iRow][iFeature], 2) + pow(dMatrixQ[iFeature][iColumn], 2)));
					}
					*/
				}
			}
		}
		if (iStep % STEP_PRINT == 0) {
			outfile << "step: " << iStep << "\tdErrorSum: " << dErrorSum << endl;
		}
		if (dErrorSum < 0.001)
		{
			break;
		}
	}
}
//*/


// test label
#include<algorithm>
void label_matrix(int *matrixA, int N)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			*(matrixA + i*N + j) = (N - i + j) % N;
		}
	}
}


int main()
{
	srand((unsigned)time(NULL));

	// sgd
#if 0
	//srand(time(NULL));
	for (int i = 0; i < ROW_NUM; ++i)
	{
		for (int j = 0; j < COLUMN_NUM; ++j)
		{
			for (int k = 0; k < K_NUM; ++k)
			{
				
				dMatrixP[i][k] = (rand() % RAND_MAX) / (double)(RAND_MAX);
				dMatrixQ[k][j] = (rand() % RAND_MAX) / (double)(RAND_MAX);
				y[k][j] = (rand() % RAND_MAX) / (double)(RAND_MAX);
			}
		}
	}
	printMatrix(dMatrixR);
	
	bool bSGD = 0;
	bool bSGDPP = 1;
	if (bSGD)
	{
		sgd(dMatrixR, dMatrixP, dMatrixQ, ROW_NUM, COLUMN_NUM, K_NUM, TEST_STEPS);
		matrixMultiply(dMatrixRNext, dMatrixP, dMatrixQ, ROW_NUM, COLUMN_NUM, K_NUM);
	}
	if (bSGDPP)
	{
		sgdPlusPlus(dMatrixR, dMatrixP, dMatrixQ, ROW_NUM, COLUMN_NUM, K_NUM, TEST_STEPS);
		double dMatrixBias[ROW_NUM][K_NUM];
		matrixAdd(dMatrixP, sumY, dMatrixBias);
		matrixMultiply(dMatrixRNext, dMatrixBias, dMatrixQ, ROW_NUM, COLUMN_NUM, K_NUM);

	}
	
	printMatrix(dMatrixRNext);
	cout << "hello world" << endl;
	outfile.close();
#endif

	// test label_matrix
#if 0
	const int N = 5;
	int *matrixA = new int[N * N];
	label_matrix(matrixA, N);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			cout << *(matrixA + i*N + j) << " ";
		}
		cout << endl;
	}
	cout << endl;
	sort(matrixA, matrixA + N*N);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			cout << *(matrixA + i*N + j) << " ";
		}
		cout << endl;
	}
#endif

	#pragma warning(disable:4996)
	string inputFile = "input.txt";
	string outputFile = "output.txt";
	freopen(inputFile.c_str(), "r", stdin);
	freopen(outputFile.c_str(), "w", stdout);

	//randomGenerateMatrix();
	//computeSubset();


	fclose(stdout);
	fclose(stdin);

	//getchar();
	return 0;
}
