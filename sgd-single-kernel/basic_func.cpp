#include "basic_func.h"
#include "parameter.h"


void readMatrix(typeRate **m)
{
	int row, column;
	cin >> row >> column;
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < column; ++j)
		{
			cin >> m[i][j];
			cout << m[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}


void printList(int *a, int len)
{
	for (int i = 0; i < len; ++i)
	{
		cout << a[i] << " ";
	}
	cout << "\n" << endl;
}


void printMatrix(typeRate **m, int rowNum, int colNum)
{
	for (int i = 0; i < rowNum; ++i)
	{
		for (int j = 0; j < colNum; ++j)
		{
			cout << m[i][j] << " ";
		}
		cout << endl;
	}
	cout << "\n" << endl;
}


void transposeMatrix(typeRate **m, int row, int column)
{
	cout << "transposeMatrix called:" << endl;
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			swap(m[i][j], m[j][i]);
		}
	}
	cout << endl;
}