#include "gpusgd_serial.h"

double **matrixRate;	// ���־��� size: M * N
double **matrixUser;	// size: K * M
double **matrixItem;	// size: K * N
int M;	// matrixRate ����
int N;	// matrixRate ����
int K;	// ��������ά��
int subBlockNumL = 64;	// subBlockNumL * subBlockNumL���ӿ�
int subBlockLen = max(M, N) / subBlockNumL;	// �ӿ��СΪ subBlockLen * subBlockLen

int **matrixPattern;	// size: subBlockNumL * subBlockNumL, �� ģʽs * �ӿ�t 
int **matrixSubset;		// size: (subBlockNumL * subBlockNumL)

// �ӿ�bid�����־����еı߽�beg��end
workset *sWorkset;	// size: (subBlockNumL * subBlockNumL)

// �ӿ�bid�б���tag��ǩ������ֵ��Ŀ
int **seg;			// size: (subBlockNumL * subBlockNumL) * subBlockLen, �� bid*tag

// �ӿ�bid��ÿ������ֵ��tag�ı߽�from��to
workseg **sWorkseg;	// size: (subBlockNumL * subBlockNumL) * subBlockLen, �� bid*tag

rateNode *rateNodeArray;
subBlock *subBlockArray;	// size: subBlockNumL * subBlockNumL���ӿ�


/********************************* BEGIN:  *********************************/
/********************************** END:  **********************************/

/********************************* BEGIN: class rateNode *********************************/


// ����rateNode�������ӿ��±�x y��bid
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

// ���ӿ�������rateNode����label
void subBlock::labelNodeInSubBlock()
{
	for (int i = 0; i < rateNum; ++i)
	{
		subBlockNodeArray[i].setLabel();
	}
}

// �����ӿ�bid������label��seg(bid, label): �ӿ�bid�б��б�ǩlabel������ֵ���������浽labelNumArray
void subBlock::computeSeg()
{
	for (int i = 0; i < rateNum; ++i)
	{
		int label = subBlockNodeArray[i].getLabel();
		++labelNumArray[label];
	}
}

// TO-DO
// ����workseg(bid, label)��from��to�����ӿ�bid������ֵlabel����ʼλ��
void subBlock::computeWorkSeg()
{
}

/********************************** END: class subBlock **********************************/


/********************************* BEGIN: class workset *********************************/
// ���õ�workset��Ա����
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

// ���ü�¼������matrixSubset�ó�ÿ��workset��beg��end
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

// �����ӿ�bid�����б�ǩ������ֵ��Ŀ�����浽seg����ĵ�bid��
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

// ���ü�¼������seg�ó�ÿ��workseg��from��to
void setWorkseg(int bid, int tag)
{
	sWorkseg[bid][tag].from = sWorkset[bid].beg;	// ����tag == 0 �����
	for (int i = 0; i <= tag - 1; ++i)
	{
		sWorkseg[bid][tag].from += seg[bid][i];
	}
	sWorkseg[bid][tag].to = sWorkseg[bid][tag].from + seg[bid][tag];
}

/********************************** END: class workset **********************************/


// ����m�ڴ����
void newMatrix(double **m, int rowNum, int colNum)
{
	m = new double *[rowNum];
	for (int i = 0; i < rowNum; ++i)
	{
		m[i] = new double[colNum];
	}
}

// ����m�ڴ�����
void deleteMatrix(double **m, int rowNum, int colNum)
{
	for (int i = 0; i < rowNum; ++i)
	{
		delete[] m[i];
	}
	delete[] m;
}

// �����ʼ������(0~1)
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
// ������shuffle
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

// �������/�� shuffle (����rowShuffle)
void randomShuffleMatrix(double **m, int rowNum, int columnNum)
{
	cout << "randomShuffleMatrix called: " << endl;

	// �б任
	transposeMatrix(m, rowNum, columnNum);
	//printMatrix(m, rowNum, columnNum);
	rowShuffle(m, rowNum, columnNum);
	//printMatrix(m, rowNum, columnNum);

	// ��ԭ
	transposeMatrix(m, rowNum, columnNum);

	// �б任
	//printMatrix(m, rowNum, columnNum);
	rowShuffle(m, rowNum, columnNum);
	//printMatrix(m, rowNum, columnNum);
	cout << endl;
}

// ����ֿ飬��ͳ��ÿ�������Ԫ�ظ���
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

// ����ӿ�bid�Ƿ�Խ��
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

// �����ӿ�b_xy��ID�������ӿ��СΪ: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockLen, int subBlockIdxX, int subBlockIdxY)
{
	return subBlockIdxX * subBlockLen + subBlockIdxY;
}

// ��¼�ӿ�b_xy��ID: bid����ά����pattern(s, t) 
// �ӿ�b_xy �ǵ�s��ģʽ�еĵ�t���ӿ�, ���computeSubBlockID(z, x, y)�Ľ������pattern(s,t)
void setPattern(int **matrixPattern, int s, int t, int subBlockLen, int x, int y)
{
	matrixPattern[s][t] = computeSubBlockID(subBlockLen, x, y);
}

// ���أ�subset(x, y)
// �ӿ� b_xy ����������ֵ����(����Ԫ��)
int computeSubset(double **m, int subBlockIdxX, int subBlockIdxY, int subBlockLen)
{
	int bid = computeSubBlockID(subBlockLen, subBlockIdxX, subBlockIdxY);
	int checkResult = checkSubBlockBoundary(bid);
	// ��ȫԽ��
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

// ��¼����subset(x, y)��ȫ�ֶ�ά����matrixSubset
// �ӿ� b_xy ����������ֵ����(����Ԫ��)
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

// ����bid��Ӧ���ӿ�x,y�±�
void getBlockXY(int bid, int &x, int &y)
{
	x = (bid / subBlockLen) + 1;
	y = bid % subBlockLen;
}

// ����Rui�������ӿ�x,y�±�
void getBlockXY(int u, int i, int &x, int &y)
{
	x = u / subBlockLen;
	y = i / subBlockLen;
}


// ���ӿ�b_xy����Ԫ��(��0)����label
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