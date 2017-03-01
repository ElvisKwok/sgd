#include "gpusgd_serial.h"

double **matrixRate;	// 评分矩阵 size: M * N
double **matrixUser;	// size: K * M
double **matrixItem;	// size: K * N
int M;	// matrixRate 行数
int N;	// matrixRate 列数
int K;	// 隐含向量维数
int subBlockNumL = 64;	// subBlockNumL * subBlockNumL个子块
int subBlockLen = max(M, N) / subBlockNumL;	// 子块大小为 subBlockLen * subBlockLen

int **matrixPattern;	// size: subBlockNumL * subBlockNumL, 即 模式s * 子块t 
int **matrixSubset;		// size: (subBlockNumL * subBlockNumL)

// 子块bid在评分矩阵中的边界beg和end
workset *sWorkset;	// size: (subBlockNumL * subBlockNumL)

// 子块bid中标有tag标签的评价值数目
int **seg;			// size: (subBlockNumL * subBlockNumL) * subBlockLen, 即 bid*tag

// 子块bid中每个评价值组tag的边界from和to
workseg **sWorkseg;	// size: (subBlockNumL * subBlockNumL) * subBlockLen, 即 bid*tag

rateNode *rateNodeArray;
subBlock *subBlockArray;	// size: subBlockNumL * subBlockNumL个子块


/********************************* BEGIN:  *********************************/
/********************************** END:  **********************************/

/********************************* BEGIN: class rateNode *********************************/


// 计算rateNode所属的子块下标x y和bid
void setSubBlockIdx(rateNode &node)
{
	node.subBlockIdxX = node.u / subBlockLen;
	node.subBlockIdxY = node.i / subBlockLen;
	node.bid = node.subBlockIdxX*subBlockLen + node.subBlockIdxY;
}


void setLabel(rateNode &node)
{
	int u = node.u;
	int i = node.i;
	int subBlockIdxX = node.subBlockIdxX;
	int subBlockIdxY = node.subBlockIdxY;
	int deltaI = u - subBlockIdxX * subBlockLen;
	int deltaJ = i - subBlockIdxY * subBlockLen;
	node.label = (subBlockLen - deltaI + deltaJ) % subBlockLen;
}

/********************************** END: class rateNode **********************************/


/********************************* BEGIN: class subBlock *********************************/
subBlock::subBlock(int blockId, int size, rateNode *nodeArray)
{
	this->bid = blockId;
	this->rateNum = size;

	if (size == 0 || nodeArray == NULL)
	{
		subBlockNodeArray = NULL;
	}
	else
	{
		subBlockNodeArray = new rateNode[size];
		for (int i = 0; i < size; ++i)
		{
			subBlockNodeArray[i] = nodeArray[i];
		}
	}

	labelNumArray = new int[subBlockLen];
	memset(labelNumArray, 0, subBlockLen*sizeof(int));

	worksegArray = new workseg[subBlockLen];
}

subBlock::~subBlock()
{
	if (subBlockNodeArray != NULL)
	{
		delete[]subBlockNodeArray;
	}

	if (labelNumArray != NULL)
	{
		delete[]labelNumArray;
	}
	if (worksegArray != NULL)
	{
		delete[]worksegArray;
	}
}

void subBlock::setBid()
{
	this->bid = subBlockIdxX*subBlockLen + subBlockIdxY;
}

int subBlock::getBid()
{
	return this->bid;
}

void subBlock::setSubBlockIdx()
{
	this->subBlockIdxX = (bid / subBlockLen) + 1;
	this->subBlockIdxY = bid % subBlockLen;
}

void subBlock::getSubBlockIdx(int &x, int &y)
{
	x = this->subBlockIdxX;
	y = this->subBlockIdxY;
}

// 对子块内所有rateNode进行label
void subBlock::labelNodeInSubBlock()
{
	for (int i = 0; i < rateNum; ++i)
	{
		subBlockNodeArray[i].setLabel();
	}
}

// 计算子块bid的所有label的seg(bid, label): 子块bid中标有标签label的评价值个数，保存到labelNumArray
void subBlock::computeSeg()
{
	for (int i = 0; i < rateNum; ++i)
	{
		int label = subBlockNodeArray[i].getLabel();
		++labelNumArray[label];
	}
}

// TO-DO
// 计算workseg(bid, label)的from和to，即子块bid中评价值label的起始位置
void subBlock::computeWorkSeg()
{
}

/********************************** END: class subBlock **********************************/


/********************************* BEGIN: class workset *********************************/
// 无用的workset成员函数
/*
void workset::setBeg(int subBlockIdxX, int subBlockIdxY)
{
	int beg = 0;
	for (int j = 0; j <= subBlockIdxY; ++j)
	{
		for (int i = 0; i < subBlockIdxX; ++i)
		{
			beg += computeSubset(matrixRate, i, j, subBlockLen);
		}
	}
	this->beg = beg;
}

void workset::setEnd(int subBlockIdxX, int subBlockIdxY)
{
	this->end = this->beg + computeSubset(matrixRate, subBlockIdxX, subBlockIdxY, subBlockLen);
}
*/

// 利用记录的数组matrixSubset得出每个workset的beg和end
void setWorkset(int bid)
{
	int subBlockIdxX;
	int subBlockIdxY;
	getBlockXY(bid, subBlockIdxX, subBlockIdxY);

	int beg = 0;
	for (int i = 0; i <= subBlockIdxX; ++i)
	{
		for (int j = 0; j < subBlockIdxY; ++j)
		{
			beg += matrixSubset[i][j];
		}
	}
	sWorkset[bid].beg = beg;
	sWorkset[bid].end = beg + matrixSubset[subBlockIdxX][subBlockIdxY];
}

// 计算子块bid中所有标签的评价值数目，保存到seg数组的第bid行
void computeSeg(int bid)
{
	memset(seg[bid], 0, subBlockLen * sizeof(int));
	int rateNum = subBlockArray[bid].rateNum;
	for (int i = 0; i < rateNum; ++i)
	{
		int label = subBlockArray[bid].subBlockNodeArray[i].label;
		++seg[bid][label];
	}
}

// 利用记录的数组seg得出每个workseg的from和to
void setWorkseg(int bid, int tag)
{
	sWorkseg[bid][tag].from = sWorkset[bid].beg;	// 包含tag == 0 的情况
	for (int i = 0; i <= tag - 1; ++i)
	{
		sWorkseg[bid][tag].from += seg[bid][i];
	}
	sWorkseg[bid][tag].to = sWorkseg[bid][tag].from + seg[bid][tag];
}

/********************************** END: class workset **********************************/


// 矩阵m内存分配
void newMatrix(double **m, int rowNum, int colNum)
{
	m = new double *[rowNum];
	for (int i = 0; i < rowNum; ++i)
	{
		m[i] = new double[colNum];
	}
}

// 矩阵m内存销毁
void deleteMatrix(double **m, int rowNum, int colNum)
{
	for (int i = 0; i < rowNum; ++i)
	{
		delete[] m[i];
	}
	delete[] m;
}

// 随机初始化矩阵(0~1)
void randomInitMatrix(double **m, int rowNum, int colNum)
{
	for (int i = 0; i < rowNum; ++i)
	{
		for (int j = 0; j < colNum; ++j)
		{
			m[i][j] = (rand() % RAND_MAX) / (double)(RAND_MAX);
		}
	}
}


/********************************* BEGIN: shuffle matrix *********************************/
// 矩阵行shuffle
void rowShuffle(double **a, int rowNum, int colNum)
{
	cout << "rowShuffle called: " << endl;

	int *newPos = new int[rowNum];
	int *permList = new int[rowNum];
	//int testList[5] = { 0, 1, 2, 3, 4 };
	for (int i = 0; i < rowNum; ++i)
	{
		newPos[i] = i;
		permList[i] = i;
	}
	random_shuffle(permList, permList + rowNum);
	//cout << "rand_perm list: ";
	//printList(permList, rowNum);
	//cout << endl;
	for (int i = 0; i < rowNum - 1; ++i)
	{
		int newPosition = newPos[permList[i]];
		for (int j = 0; j < colNum; ++j)
		{

			//swap(testList[i], testList[newPosition]);
			swap(a[i][j], a[newPosition][j]);
		}
		newPos[i] = permList[i];
	}
	//printList(testList, rowNum);
	cout << endl;
	delete newPos;
	delete permList;
}

// 矩阵的行/列 shuffle (调用rowShuffle)
void randomShuffleMatrix(double **m, int rowNum, int columnNum)
{
	cout << "randomShuffleMatrix called: " << endl;

	// 列变换
	transposeMatrix(m, rowNum, columnNum);
	//printMatrix(m, rowNum, columnNum);
	rowShuffle(m, rowNum, columnNum);
	//printMatrix(m, rowNum, columnNum);

	// 复原
	transposeMatrix(m, rowNum, columnNum);

	// 行变换
	//printMatrix(m, rowNum, columnNum);
	rowShuffle(m, rowNum, columnNum);
	//printMatrix(m, rowNum, columnNum);
	cout << endl;
}

// 矩阵分块，并统计每块包含的元素个数
int blockMatrix(double **a, int rowNums, int blockLen)
{
	cout << "blockMatrix called:" << endl;
	const int blockSize = blockLen;	// element_num_per_block = blockSize * blockSize;
	const int divider = rowNums / blockSize;
	int *b = new int[divider*divider];
	memset(b, 0, divider*divider*sizeof(int));

	// block
	for (int i = 0; i < rowNums; ++i)
	{
		for (int j = 0; j < rowNums; ++j)
		{
			//cout << a[i][j] << " ";
			if (a[i][j] != 0)
			{
				int bi = i / blockSize;
				int bj = j / blockSize;
				++b[bi * divider + bj];
			}
		}
		//cout << endl;
	}

	// count
	int max = INT_MIN;
	int min = INT_MAX;
	int avg = 0;
	//cout << endl;
	for (int i = 0; i < divider; ++i)
	{
		for (int j = 0; j < divider; ++j)
		{
			cout << b[i * divider + j] << " ";
			max = (max > b[i * divider + j]) ? max : b[i * divider + j];
			min = (min < b[i * divider + j]) ? min : b[i * divider + j];
			avg += b[i * divider + j];
		}
		cout << endl;
	}

	cout << "max: " << max << endl;
	cout << "min: " << min << endl;
	cout << "avg: " << (avg / (divider*divider)) << endl;

	cout << "diff: " << (max - min) << endl;

	delete[]b;
	return (max - min);
}

// testShuffle
void randomGenerateMatrix()
{
	srand((unsigned)time(NULL));
	const int MAX = 1000;	// element_num_all = MAX * MAX;
	double **a = new double*[MAX];
	int nnz = 0;
	for (int i = 0; i < MAX; ++i)
	{
		a[i] = new double[MAX];
		for (int j = 0; j < MAX; ++j)
		{
			if (rand() % 100 == 0)
			{
				a[i][j] = (rand() % 5) + 1;	// rating: 1 ~ 5
				++nnz;
			}
			else
			{
				a[i][j] = 0;
			}
		}
	}

	cout << "before shuffle, block result: " << endl;
	blockMatrix(a, MAX, 50);

	int min_diff = INT_MAX;
	for (int i = 0; i < 10; ++i)
	{
		randomShuffleMatrix(a, MAX, MAX);

		cout << "after shuffle, block result: " << endl;
		int diff = blockMatrix(a, MAX, 50);
		min_diff = (min_diff < diff) ? min_diff : diff;
	}
	cout << "min_diff: " << min_diff << endl;

	/*
	int max = INT_MIN;
	int min = INT_MAX;
	for (int i = 0; i < 100; ++i)
	{
	randomShuffleMatrix(a, MAX);

	int diff = blockMatrix(a, MAX);
	max = (diff > max) ? diff : max;
	min = (diff < max) ? diff : max;

	}
	*/

	//cout << max << "\t" << min << endl;

	for (int i = 0; i < MAX; ++i)
	{
		delete[] a[i];
	}
	delete[]a;
}
/********************************** END: shuffle matrix **********************************/

// 检查子块bid是否越界
int checkSubBlockBoundary(int bid)
{
	int subBlockIdxX;
	int subBlockIdxY;
	getBlockXY(bid, subBlockIdxX, subBlockIdxY);
	
	if ((subBlockIdxX * subBlockLen >= M) || (subBlockIdxY * subBlockLen >= N))
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

// 计算子块b_xy的ID，其中子块大小为: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockLen, int subBlockIdxX, int subBlockIdxY)
{
	return subBlockIdxX * subBlockLen + subBlockIdxY;
}

// 记录子块b_xy的ID: bid到二维数组pattern(s, t) 
// 子块b_xy 是第s种模式中的第t个子块, 则把computeSubBlockID(z, x, y)的结果放入pattern(s,t)
void setPattern(int **matrixPattern, int s, int t, int subBlockLen, int x, int y)
{
	matrixPattern[s][t] = computeSubBlockID(subBlockLen, x, y);
}

// 返回：subset(x, y)
// 子块 b_xy 包含的评价值个数(非零元素)
int computeSubset(double **m, int subBlockIdxX, int subBlockIdxY, int subBlockLen)
{
	int bid = computeSubBlockID(subBlockLen, subBlockIdxX, subBlockIdxY);
	int checkResult = checkSubBlockBoundary(bid);
	// 完全越界
	if (checkResult == 1)
	{
		return 0;
	}

	int cnt = 0;
	int leftUpIdxRow = subBlockIdxX * subBlockLen;
	int leftUpIdxCol = subBlockIdxY * subBlockLen;
	int rightDownIdxRow = leftUpIdxRow + subBlockLen - 1;
	int rightDownIdxCol = leftUpIdxCol + subBlockLen - 1;
	for (int i = leftUpIdxRow; i <= min(rightDownIdxRow, M-1); ++i)
	{
		for (int j = leftUpIdxCol; j <= min(rightDownIdxCol, N-1); ++j)
		{
			if (m[i][j] != 0) // fabs(m[i][j]) > 1e-8
			{
				++cnt;
			}
		}
	}
	return cnt;
}

// 记录所有subset(x, y)到全局二维数组matrixSubset
// 子块 b_xy 包含的评价值个数(非零元素)
void computeAllSubset()
{
	int cnt = 0;
	for (int subBlockIdxX = 0; subBlockIdxX < subBlockNumL; ++subBlockIdxX)
	{
		for (int subBlockIdxY = 0; subBlockIdxY < subBlockNumL; ++subBlockIdxY)
		{
			/*
			cnt += computeSubset(matrixRate, subBlockIdxX, subBlockIdxY, subBlockLen);
			matrixSubset[subBlockIdxX][subBlockIdxY] = cnt;
			*/
			matrixSubset[subBlockIdxX][subBlockIdxY] = computeSubset(matrixRate, subBlockIdxX, subBlockIdxY, subBlockLen);
		}
	}
}

// 计算bid对应的子块x,y下标
void getBlockXY(int bid, int &x, int &y)
{
	x = (bid / subBlockLen) + 1;
	y = bid % subBlockLen;
}

// 计算Rui所属的子块x,y下标
void getBlockXY(int u, int i, int &x, int &y)
{
	x = u / subBlockLen;
	y = i / subBlockLen;
}


// 对子块b_xy所有元素(非0)进行label
// TO-DO:
/*
void labelSubBlock(int subBlockIdxX, int subBlockIdxY)
{
	int leftUpIdxRow = subBlockIdxX * subBlockLen;
	int leftUpIdxCol = subBlockIdxY * subBlockLen;
	int rightDownIdxRow = leftUpIdxRow + subBlockLen - 1;
	int rightDownIdxCol = leftUpIdxCol + subBlockLen - 1;
	for (int i = 0; leftUpIdxRow + i <= rightDownIdxRow; ++i)
	{
		for (int j = 0; leftUpIdxCol + j <= rightDownIdxCol; ++j)
		{
			if (matrixRate[leftUpIdxRow + i][leftUpIdxCol + j] != 0) // fabs(matrixRate[leftUpIdxRow + i][leftUpIdxCol + j]) > 1e-8
			{
				int label = (subBlockLen - i + j) % subBlockLen;

			}
		}
	}
}
*/