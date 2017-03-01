#ifndef __BASIC_FUNC_H_
#define ___BASIC_FUNC_H_


#include <iostream>
#include <vector>
#include <string>
using namespace std;

// read matrix from input file(or stdin)
void readMatrix(double **m);

void printList(int *a, int len);

void printMatrix(double **m, int rowNum, int colNum);

void transposeMatrix(double **m, int row, int column);

#endif