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

#include "sgd.h"

using namespace std;



// test label
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


void callGPU()
{
	int size = 20;
	int *a = new int[size];
	int *b = new int[size];
	int *c = new int[size];

	for (int i = 0; i < size; ++i)
	{
		a[i] = 1;
		b[i] = 2;
	}

	solveByGPU(a, b, c, size);

	
	printList(c, size);
	delete[]a;
	delete[]b;
	delete[]c;
	getchar();
}

void test()
{
	string outputFile = "output/console_output.txt";
	freopen(outputFile.c_str(), "w", stdout);
	unitTest();
	//getchar();
	fclose(stdout);
}


int main()
{
	srand((unsigned)time(NULL));


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

	
#if 0
	string inputFile = "input.txt";
	string outputFile = "output.txt";
	freopen(inputFile.c_str(), "r", stdin);
	freopen(outputFile.c_str(), "w", stdout);
#endif
	//randomGenerateMatrix();
	//computeSubset();

	//callGPU();
	test();

	//fclose(stdout);
	//fclose(stdin);

	//getchar();
	return 0;
}

