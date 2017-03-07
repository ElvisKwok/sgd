#include "gpusgd_serial.h"

string parameterFile = "parameter.txt";
string inputFile = "input.txt";
string resultFile = "predict_result.txt";
string modelFile = "model.txt";

int MAX_ITER;				// ����������
double lambda;				// ����ϵ��
double gamma;				// ѧϰ��
int K;						// ��������ά��
int subBlockNumL = 64;		// subBlockNumL * subBlockNumL���ӿ�
int threads_per_block;		// һ��block�������߳���

// DELETE
typeRate **matrixRate;		// ���־��� size: M * N
typeRate **matrixUser;		// size: M * K
typeRate **matrixItem;		// size: N * K
int M;						// matrixRate ����
int N;						// matrixRate ����
int subBlockNum;			// �ӿ�����Ŀ, value = subBlockNumL * subBlockNumL
int subBlockLen;			// �ӿ��СΪ subBlockLen * subBlockLen, value = max(M, N) / subBlockNumL
int subBlockNodeNum;		// �ӿ��Сsize������0���0��, value = subBlockLen * subBlockLen

int NNZ;					// ����Ԫ�ظ����������ļ�(user, item, rate)������

int **matrixPattern;		// size: subBlockNumL * subBlockNumL, �� ģʽs * �ӿ�t (ע���ʼ��Ϊ-1)
// DELETE
int **matrixSubset;			// size: (subBlockNumL * subBlockNumL)
int *subsetArray;			// ��¼ÿ���ӿ�bid��nnz, size: (subBlockNumL * subBlockNumL), bidΪ�±� ������ͳ�ƿ����ȣ�

// �ӿ�bid�����־����еı߽�beg��end
sWorkset *worksetArray;		// size: (subBlockNumL * subBlockNumL)

// DELETE
// �ӿ�bid�б���tag��ǩ������ֵ��Ŀ
int **mSeg;					// size: (subBlockNumL * subBlockNumL) * subBlockLen, �� bid*tag

// �ӿ�bid��ÿ������ֵ��tag�ı߽�from��to
sWorkseg **mWorkseg;		// size: (subBlockNumL * subBlockNumL) * subBlockLen, �� bid*tag


vector<sRateNode> rateNodeVector;	// ���ڶ�̬��ȡ����
sRateNode *rateNodeArray;
sSubBlock *subBlockArray;	// size: subBlockNumL * subBlockNumL���ӿ�



/********************************* BEGIN:  File Process *********************************/
// ��ȡ�����ļ���
// M, N
// (user, item, rate)
void readFile(string fileName)
{
	initBlockDimension();

	// serial read
	/*
	ifstream inputFile(fileName);

	while (!inputFile.eof())
	{
		int userIdx;
		int itemIdx;
		typeRate rate;
		inputFile >> userIdx >> itemIdx >> rate;
		sRateNode node(userIdx, itemIdx, rate);
		setSubBlockIdx(node);
		setLabel(node);

		rateNodeVector.push_back(node);
	}
	*/

	// parallel read
	parallelReadFile(fileName, rateNodeVector);

	NNZ = rateNodeVector.size();
	rateNodeArray = &rateNodeVector[0];
}


// debug:
void printRateNode(sRateNode &node)
{
	printVar("node.u", node.u);
	printVar("node.i", node.i);
	printVar("node.rate", node.rate);
	printVar("node.bid ", node.bid);
	printVar("node.subBlockIdxX", node.subBlockIdxX);
	printVar("node.subBlockIdxY ", node.subBlockIdxY);
	printVar("node.label", node.label);
}

// debug:
void unitTest()
{
	initParameter();
	readFile(inputFile);
	sortRateNodeArrayBid();
	for (int i = 0; i < rateNodeVector.size(); ++i)
	{
		printRateNode(rateNodeArray[i]);
		printLine();
	}
}


// ���ѵ��ģ��(matrixUser, matrixItem)
void writeFile(string fileName)
{
	ofstream outFile(fileName);

	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			outFile << matrixUser[i][k] << " ";
		}
		outFile << endl;
	}

	for (int i = 0; i < N; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			outFile << matrixItem[i][k] << " ";
		}
		outFile << endl;
	}
}

// TO-DO
// ���Ԥ����
// resultArray��user_item�����˳��
void writeFile(typeRate *resultArray, int predictNum, string fileName)
{
	ofstream outFile(fileName);
	for (int i = 0; i < predictNum; ++i)
	{
		outFile << resultArray[i] << endl;
	}
}

// ���ݲ����ļ��趨����ʼ������
void initParameter()
{
	cout << "reading parameters from parameters.txt..." << endl;

	freopen(parameterFile.c_str(), "r", stdin);

	scanf("MAX_ITER = %d\n", &MAX_ITER);
	scanf("lambda = %lf\n", &lambda);
	scanf("gamma = %lf\n", &gamma);
	scanf("M = %d\n", &M);
	scanf("N = %d\n", &N);
	scanf("K = %d\n", &K);
	scanf("subBlockNumL = %d\n", &subBlockNumL);
	scanf("threads_per_block = %d\n", &threads_per_block);

	fclose(stdin);
	freopen("CON", "r", stdin);   //"CON"�������̨

	///*
	// test
	cout << "MAX_ITER = " << MAX_ITER << endl;
	cout << "lambda = " << lambda << endl;
	cout << "gamma = " << gamma << endl;
	cout << "M = " << M << endl;
	cout << "N = " << N << endl;
	cout << "K = " << K << endl;
	cout << "subBlockNumL = " << subBlockNumL << endl;
	cout << "threads_per_block = " << threads_per_block << endl;
	cout << endl;
	//*/

	//getchar();
}

// ���������M, N ���Ի�subBlockά�Ȳ���
void initBlockDimension()
{
	subBlockNum = subBlockNumL * subBlockNumL;		// �ӿ�����Ŀ
	subBlockLen = max(M, N) / subBlockNumL;			// �ӿ��СΪ subBlockLen * subBlockLen
	subBlockNodeNum = subBlockLen * subBlockLen;	// �ӿ��Сsize������0���0��

	// debug:
	printVar("subBlockNum", subBlockNum);
	printVar("subBlockLen", subBlockLen);
	printVar("subBlockNodeNum", subBlockNodeNum);
	printLine();
}


/********************************** END:  File Process **********************************/

/********************************* BEGIN: class sRateNode *********************************/
// ����rateNode�������ӿ��±�x y��bid
void setSubBlockIdx(sRateNode &node)
{
	node.subBlockIdxX = node.u / subBlockLen;
	node.subBlockIdxY = node.i / subBlockLen;
	node.bid = node.subBlockIdxX*subBlockLen + node.subBlockIdxY;
}

void setLabel(sRateNode &node)
{
	int u = node.u;
	int i = node.i;
	int subBlockIdxX = node.subBlockIdxX;
	int subBlockIdxY = node.subBlockIdxY;
	int deltaI = u - subBlockIdxX * subBlockLen;
	int deltaJ = i - subBlockIdxY * subBlockLen;
	node.label = (subBlockLen - deltaI + deltaJ) % subBlockLen;
}

// sort�Ƚ�ν��pred
bool compare_bid_label(sRateNode a, sRateNode b)
{
	// return a.bid < b.bid; // ֻ��bid����
	if (a.bid < b.bid)
	{
		return 1;
	}
	if (a.bid == b.bid && a.label < b.label)
	{
		return 1;
	}
	return 0;
}

// ����bid��rateNodeArray��������
void sortRateNodeArrayBid()
{
	sort(rateNodeArray, rateNodeArray + NNZ, compare_bid_label);
}
/********************************** END: class sRateNode **********************************/


/********************************* BEGIN: class sSubBlock *********************************/
// DELETE
// setSubBlockIdx()
// setRateNum() 
// allocSubBlockNodeArray() 
// labelNodeInSubBlock()
#if 0
// ����bid�����ӿ�x y����
void setSubBlockIdx(sSubBlock &subBlock)
{
	subBlock.subBlockIdxX = (subBlock.bid / subBlockLen) + 1;
	subBlock.subBlockIdxY = subBlock.bid % subBlockLen;
}

// �����ӿ�����ķ�0Ԫ�ظ���
void setRateNum(sSubBlock &subBlock)
{
	subBlock.rateNum = computeSubset(subBlock.subBlockIdxX, subBlock.subBlockIdxY);
}

// DELETE
// ȡ����ֱ�Ӹ���ָ�룬��Ҫ�ظ�����ռ�
/*
// �����ӿ��node����ռ�
void allocSubBlockNodeArray(sSubBlock &subBlock)
{
	subBlock.subBlockNodeArray = new sRateNode[subBlock.rateNum];
}
*/

// ���ӿ�������rateNode����label
void labelNodeInSubBlock(sSubBlock &subBlock)
{
	for (int i = 0; i < subBlock.rateNum; ++i)
	{
		setLabel(subBlock.subBlockNodeArray[i]);
	}
}

#endif


// DELETE
// workseg����
/*
// �����ӿ�bid������label��ǩ������ֵ��Ŀseg(bid, label)�����浽mSeg����ĵ�bid��
void computeSeg(int bid)
{
	memset(mSeg[bid], 0, subBlockLen * sizeof(int));
	int rateNum = subBlockArray[bid].rateNum;
	for (int i = 0; i < rateNum; ++i)
	{
		int label = subBlockArray[bid].subBlockNodeArray[i].label;
		++mSeg[bid][label];
	}
}

// ���ü�¼������seg�ó�ÿ��workseg��from��to
// ����workseg(bid, label)��from��to�����ӿ�bid������ֵlabel����ʼλ��
void computeWorkseg(int bid, int tag)
{
	mWorkseg[bid][tag].from = worksetArray[bid].beg;	// ����tag == 0 �����
	for (int i = 0; i <= tag - 1; ++i)
	{
		mWorkseg[bid][tag].from += mSeg[bid][i];
	}
	mWorkseg[bid][tag].to = mWorkseg[bid][tag].from + mSeg[bid][tag];
}
*/

/*
// �����ӿ�������pattern
void setSubBlockPattern(sSubBlock &subBlock)
{
	int x = subBlock.subBlockIdxX;
	int y = subBlock.subBlockIdxY;
	subBlock.pattern = (subBlockNumL - x + y) % subBlockNumL;
}

// ���������ӿ��pattern����ά����pattern
void setAllPattern()
{
	vector<int> cnt(subBlockNumL, 0);
	for (int i = 0; i < subBlockNum; ++i)
	{
		int pattern = subBlockArray[i].pattern;
		int t = cnt[pattern]++;		// ��sģʽ�ĵ�t���ӿ�
		int bid = subBlockArray[i].bid;
		setPattern(pattern, t, bid);
	}
}
*/

// ���������ӿ��pattern����ά����pattern
void setAllPattern()
{
	int x, y, pattern;
	vector<int> cnt(subBlockNumL, 0);
	for (int bid = 0; bid < subBlockNum; ++bid)
	{
		if (subsetArray[bid] > 0)
		{
			getBlockXY(bid, x, y);
			pattern = (subBlockNumL - x + y) % subBlockNumL;
			int t = cnt[pattern]++;		// ��sģʽ�ĵ�t���ӿ�
			setPattern(pattern, t, bid);
		}
	}
}

// DELETE
// �ӿ���rateNode����
/*
bool compare_label(sRateNode a, sRateNode b)
{
	return a.label < b.label;
}

// ����label���ӿ�������rateNode��������
void sortLabelInSubBlock(sSubBlock &subBlock)
{
	sort(subBlock.subBlockNodeArray, subBlock.subBlockNodeArray + subBlock.rateNum, compare_label);
}

// ����sortLabelInSubBlock���������ӿ����label����
void sortLabelAll()
{
	for (int i = 0; i < subBlockNum; ++i)
	{
		sortLabelInSubBlock(subBlockArray[i]);
	}
}
*/

// DELETE
// ȡ����ֱ�Ӹ���ָ�룬��Ҫ�ظ�����ռ�
/*
// �ͷ��ӿ�Ķ�̬�ڴ�ռ�
void destroySubBlock(sSubBlock &subBlock)
{
	delete[]subBlock.subBlockNodeArray;
}
*/
/********************************** END: class subBlock **********************************/


/********************************* BEGIN: class sWorkset sWorkseg *********************************/
// DELETE
// ���õ�sWorkset��Ա����
/*
void sWorkset::setBeg(int subBlockIdxX, int subBlockIdxY)
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

void sWorkset::setEnd(int subBlockIdxX, int subBlockIdxY)
{
	this->end = this->beg + computeSubset(matrixRate, subBlockIdxX, subBlockIdxY, subBlockLen);
}
*/

// DELETE
// ���ü�¼������matrixSubset�ó�����workset��beg��end
/*
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
	worksetArray[bid].beg = beg;
	worksetArray[bid].end = beg + matrixSubset[subBlockIdxX][subBlockIdxY];
}
*/

// NEW
// ɨ��bid������rateNodeArray, �ó�ÿ��workset��beg��end�����浽worksetArray
// ͬʱͳ��ÿ��subset��ֵ,���浽subsetArray
void setWorkset()
{
	if (NNZ <= 0)
	{
		return;
	}

	int beg = 0;
	int curBid = 0;
	for (int i = 0; i < NNZ; ++i)
	{
		curBid = rateNodeArray[i].bid;
		beg = i;
		while (i < NNZ && (curBid == rateNodeArray[i].bid))	// ����ִ��һ��
		{
			++i;
		}
		worksetArray[curBid].beg = beg;
		worksetArray[curBid].end = i;
		subsetArray[curBid] = i - beg;
		--i; // forѭ����++i
	}
}

// ɨ��bid_label������rateNodeArray, �ó�ÿ��workseg��from��to�����浽mWorkseg
void setWorkseg()
{
	if (NNZ <= 0)
	{
		return;
	}

	int beg = 0;
	int end = 0;
	int from = 0;
	int to = 0;
	int curLabel;
	for (int curBid = 0; curBid < subBlockNum; ++curBid)	// ÿ���ӿ�curBid
	{
		beg = worksetArray[curBid].beg;
		end = worksetArray[curBid].end;
		
		for (int i = beg; i < end; ++i)						// �ӿ���
		{
			curLabel = rateNodeArray[i].label;
			from = i;
			while (i < NNZ && (curLabel == rateNodeArray[i].label))	// ����ִ��һ��
			{
				++i;
			}
			mWorkseg[curBid][curLabel].from = from;
			mWorkseg[curBid][curLabel].to = i;
			--i; // forѭ����++i
		}
	}
}
/********************************** END: class sWorkset sWorkseg **********************************/


// ����m�ڴ����
void newMatrix(typeRate **m, int rowNum, int colNum)
{
	m = new typeRate *[rowNum];
	for (int i = 0; i < rowNum; ++i)
	{
		m[i] = new typeRate[colNum];
	}
}

// ����m�ڴ�����
void deleteMatrix(typeRate **m, int rowNum, int colNum)
{
	for (int i = 0; i < rowNum; ++i)
	{
		delete[] m[i];
	}
	delete[] m;
}

// �����ʼ������(0~1)
void randomInitMatrix(typeRate **m, int rowNum, int colNum)
{
	for (int i = 0; i < rowNum; ++i)
	{
		for (int j = 0; j < colNum; ++j)
		{
			m[i][j] = (rand() % RAND_MAX) / (typeRate)(RAND_MAX);
		}
	}
}


/********************************* BEGIN: shuffle matrix *********************************/
// ������shuffle
void rowShuffle(typeRate **a, int rowNum, int colNum)
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
void randomShuffleMatrix(typeRate **m, int rowNum, int columnNum)
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
int blockMatrix(typeRate **a, int rowNums, int blockLen)
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
	typeRate **a = new typeRate*[MAX];
	int nnz = 0;
	for (int i = 0; i < MAX; ++i)
	{
		a[i] = new typeRate[MAX];
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

// DELETE
// ����ӿ�bid�Ƿ�Խ��
/*
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
*/

// �����ӿ�x, y��Ӧ��bid�������ӿ��СΪ: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockLen, int subBlockIdxX, int subBlockIdxY)
{
	return subBlockIdxX * subBlockLen + subBlockIdxY;
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


// ��¼�ӿ�b_xy��ID: bid����ά����pattern(s, t) 
// �ӿ�b_xy �ǵ�s��ģʽ�еĵ�t���ӿ�, ���computeSubBlockID(z, x, y)�Ľ������pattern(s,t)
void setPattern(int s, int t, int x, int y)
{
	matrixPattern[s][t] = computeSubBlockID(subBlockLen, x, y);
}

// ��¼�ӿ�b_xy��ID: bid����ά����pattern(s, t) 
// �ӿ�bid�ǵ�s��ģʽ�еĵ�t���ӿ�, ���bid����pattern(s,t)
void setPattern(int s, int t, int bid)
{
	matrixPattern[s][t] = bid;
}

// DELETE
// ԭsubset����ļ���
#if 0
// ���أ�subset(x, y)
// �ӿ� b_xy ����������ֵ����(����Ԫ��)
int computeSubset(int subBlockIdxX, int subBlockIdxY)
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
			if (matrixRate[i][j] != 0) // fabs(m[i][j]) > 1e-8
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
			matrixSubset[subBlockIdxX][subBlockIdxY] = computeSubset(subBlockIdxX, subBlockIdxY);
		}
	}
}
#endif


// ���ӿ�b_xy����Ԫ��(��0)����label
// DELETE:
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





// ��ִ�к���
void execute()
{
	// ���Ի������롢�ڴ����(array matrix����ʱmemsetΪ0)
	readFile(inputFile); // ���־���
	// memset(matrixPattern, -1, subBlockNum*sizeof(int));

	// CPU����
	
	// ����GPU����

	// ������ (model)

	// ���ִ�з�ʽΪ��Ԥ��
	// Ԥ�Ⲣ������
}