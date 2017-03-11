#include "gpusgd_serial.h"
#include "sgd.h"

string parameterFile = "input/parameter.txt";
//string inputFile = "input/ra.test_awk";
//string inputFile = "input/input.txt";
string inputFile = "input/10by10.txt";
string resultFile = "output/predict_result.txt";
string modelFile = "output/model.txt";


int MAX_ITER;				// ����������
int MAX_SHUFFLE;			// ������任����
double lambda;				// ����ϵ��
double gamma;				// ѧϰ��
int K;						// ��������ά��
int subBlockNumL = 64;		// subBlockNumL * subBlockNumL���ӿ�
int threads_per_block;		// һ��block�������߳���

// DELETE
typeRate **matrixRate;		// ���־��� size: M * N

typeRate *matrixUser;		// size: M * K
typeRate *matrixItem;		// size: N * K
int M;						// matrixRate ����
int N;						// matrixRate ����
int subBlockNum;			// �ӿ�����Ŀ, value = subBlockNumL * subBlockNumL
int subBlockLen;			// �ӿ��СΪ subBlockLen * subBlockLen, value = max(M, N) / subBlockNumL
int subBlockNodeNum;		// �ӿ��Сsize������0���0��, value = subBlockLen * subBlockLen

int NNZ;					// ����Ԫ�ظ����������ļ�(user, item, rate)������

int *matrixPattern;		// size: subBlockNumL * subBlockNumL, �� ģʽs * �ӿ�t (ע���ʼ��Ϊ-1)
// DELETE
int **matrixSubset;			// size: (subBlockNumL * subBlockNumL)
int *subsetArray;			// ��¼ÿ���ӿ�bid��nnz, size: subBlockNum, bidΪ�±� ������ͳ�ƿ����ȣ�

// �ӿ�bid�����־����еı߽�beg��end
sWorkset *worksetArray;		// size: subBlockNum

// DELETE
// �ӿ�bid�б���tag��ǩ������ֵ��Ŀ
int **mSeg;					// size: subBlockNum * subBlockLen, �� bid*tag

// �ӿ�bid��ÿ������ֵ��tag�ı߽�from��to
sWorkseg *mWorkseg;		// size: subBlockNum * subBlockLen, �� bid*tag


vector<sRateNode> rateNodeVector;	// ���ڶ�̬��ȡ����
sRateNode *rateNodeArray;

// DELETE ?
sSubBlock *subBlockArray;	// size: subBlockNum���ӿ�

vector<int> permRow;		// �������յ��б任���ԣ�size: M (ע��ֵ�ķ�Χ��1~M��������0~M-1)
vector<int> permColumn;		// �������յ��б任���ԣ�size: N



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
	int userIdx;
	int itemIdx;
	typeRate rate;
	while (!inputFile.eof())
	{
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
void printMatrixPattern()
{
	cout << "\nprintMatrixPattern: " << endl;
	for (int i = 0; i < subBlockNumL; ++i)
	{
		for (int j = 0; j < subBlockNumL; ++j)
		{
			cout << (matrixPattern + i*subBlockNumL)[j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

// debug:
void printWorksetArray()
{
	cout << "printWorksetArray: " << endl;
	for (int i = 0; i < subBlockNum; ++i)
	{
		cout << "[" << worksetArray[i].beg << ", " << worksetArray[i].end << ")\t";
	}
	cout << '\n' << endl;
}

// debug:
void printMatrixWorkseg()
{
	cout << "printMatrixWorkseg: " << endl;
	for (int i = 0; i < subBlockNum; ++i)
	{
		for (int j = 0; j < subBlockLen; ++j)
		{
			cout << "[" << (mWorkseg + i*subBlockLen)[j].from << ", " << (mWorkseg + i*subBlockLen)[j].to << ")\t";
		}
		cout << endl;
	}
	cout << endl;
}


// ���ѵ��ģ��(matrixUser, matrixItem)
void writeFile(string fileName)
{
	ofstream outFile(fileName);

	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			outFile << *(matrixUser+i*K+k) << " ";
		}
		outFile << endl;
	}
	outFile << endl;

	for (int i = 0; i < N; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			outFile << *(matrixItem + i*K + k) << " ";
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
	scanf("MAX_SHUFFLE = %d\n", &MAX_SHUFFLE);
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
	cout << "MAX_SHUFFLE = " << MAX_SHUFFLE << endl;
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
	int maxDimension = max(M, N);
	subBlockLen = maxDimension / subBlockNumL + (maxDimension % subBlockNumL > 0);	// �ӿ��СΪ subBlockLen * subBlockLen
	subBlockNodeNum = subBlockLen * subBlockLen;	// �ӿ��Сsize������0���0��

	// debug:
	printVar("subBlockNum", subBlockNum);
	printVar("subBlockLen", subBlockLen);
	//printVar("subBlockNodeNum", subBlockNodeNum);
	printLine();
}

// ��ʼ���������ݣ�ָ�룩������ռ䣬����newArray, newMatrix
void initAllData()
{
	// TO-DO

	// �����ʼ��
	//typeRate *matrixUser;		// size: M * K
	//typeRate *matrixItem;		// size: N * K
	newMatrixRandom1D(matrixUser, M, K);
	newMatrixRandom1D(matrixItem, N, K);

	//int *matrixPattern;		// size: subBlockNumL * subBlockNumL, �� ģʽs * �ӿ�t (ע���ʼ��Ϊ-1)
	newArray(matrixPattern, subBlockNum, -1);

	//int *subsetArray;			// ��¼ÿ���ӿ�bid��nnz, size: (subBlockNumL * subBlockNumL), bidΪ�±� ������ͳ�ƿ����ȣ�
	newArray(subsetArray, subBlockNum, 0);

	//sWorkset *worksetArray;		// �ӿ�bid�����־����еı߽�beg��end, size: (subBlockNumL * subBlockNumL)
	newArray(worksetArray, subBlockNum, 0);

	//sWorkseg *mWorkseg;		// �ӿ�bid��ÿ������ֵ��tag�ı߽�from��to, size: (subBlockNumL * subBlockNumL) * subBlockLen, �� bid*tag
	newArray(mWorkseg, subBlockNum * subBlockLen, 0);

	// ��ʼ��permRow, permColumn
	int i;
	for (i = 0; (i < M) && (i < N); ++i)
	{
		permRow.push_back(i+1);
		permColumn.push_back(i+1);
	}
	while (i < M)
	{
		permRow.push_back(i+1);
		++i;
	}
	while (i < N)
	{
		permColumn.push_back(i+1);
		++i;
	}
}

/********************************** END:  File Process **********************************/

/********************************* BEGIN: class sRateNode *********************************/
// ����rateNode�������ӿ��±�x y��bid
void setSubBlockIdx(sRateNode &node)
{
	node.subBlockIdxX = (node.u - 1) / subBlockLen;		// userIdx��һ��ʼ
	node.subBlockIdxY = (node.i - 1) / subBlockLen;
	node.bid = node.subBlockIdxX*subBlockNumL + node.subBlockIdxY;
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

// ����任��Ҫ���¼���bid(�任��)
void resetAllNode_blockIdx()
{
	for (int i = 0; i < NNZ; ++i)
	{
		setSubBlockIdx(rateNodeArray[i]);
	}
}

// ����任��Ҫ���¼���bid��label��final��
void resetAllNode_blockIdx_label()
{
	for (int i = 0; i < NNZ; ++i)
	{
		setSubBlockIdx(rateNodeArray[i]);
		setLabel(rateNodeArray[i]);
	}
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

// sort�Ƚ�ν��pred
bool compare_user_item(sRateNode a, sRateNode b)
{
	// return a.bid < b.bid; // ֻ��bid����
	if (a.u < b.u)
	{
		return 1;
	}
	if (a.u == b.u && a.i < b.i)
	{
		return 1;
	}
	return 0;
}

// ����user_item��rateNodeArray��������
void sortRateNodeArrayUserItem()
{
	sort(rateNodeArray, rateNodeArray + NNZ, compare_user_item);
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
			//debug:
			/*
			printVar("pattern", pattern);
			printVar("t", t);
			printVar("bid", bid);
			*/
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
			(mWorkseg + curBid*subBlockLen)[curLabel].from = from;		// .�����ȼ���*�ߣ���*Ҫ������ (*(*(mWorkseg + curBid*subBlockLen + curLabel))).from
			(mWorkseg + curBid*subBlockLen)[curLabel].to = i;
			
			//mWorkseg[curBid][curLabel].from = from;
			//mWorkseg[curBid][curLabel].to = i;
			--i; // forѭ����++i
		}
	}
}
/********************************** END: class sWorkset sWorkseg **********************************/


// ����array�ڴ����
template <typename T>
void newArray(T* &array, int n, int val)
{
	array = new T[n];
	memset(array, val, n * sizeof(T));
}

// ����m�ڴ����
template <typename T>
void newMatrix(T** &m, int rowNum, int colNum, int val)
{
	m = new T *[rowNum];
	for (int i = 0; i < rowNum; ++i)
	{
		m[i] = new T[colNum];
		memset(m[i], val, colNum * sizeof(T));
	}
}

// 1ά�ľ���m�ڴ����, �������ʼ��
template <typename T>
void newMatrixRandom1D(T* &m, int rowNum, int colNum)
{
	m = new T[rowNum * colNum];
	for (int i = 0; i < rowNum; ++i)
	{
		for (int j = 0; j < colNum; ++j)
		{
			*(m + i * colNum + j) = (rand() % RAND_MAX) / (typeRate)(RAND_MAX);
		}
	}
	
}

// ����array�ڴ�����
template <typename T>
void deleteArray(T* &array, int n, int val)
{
	delete []array;
}

// ����m�ڴ�����
template <typename T>
void deleteMatrix(T** &m, int rowNum, int colNum)
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
// debug:
void printNodeArrayAsMatrix()
{
	sortRateNodeArrayUserItem();
	int curUser;
	for (int i = 0; i < NNZ; ++i)
	{

		curUser = rateNodeArray[i].u;
		cout << curUser << ": ";
		while (i < NNZ && (curUser == rateNodeArray[i].u))	// ����ִ��һ��
		{
			cout << rateNodeArray[i].i << " ";
			++i;
		}
		cout << endl;
		--i;
	}
}

// ����ӿ�nnz��max_diff
int getMaxDiff()
{
	int max = *max_element(subsetArray, subsetArray + subBlockNum);
	int min = *min_element(subsetArray, subsetArray + subBlockNum);
	//printList(subsetArray, subBlockNum);
	return max - min;
}

// ����ӿ���ȶ�
double getEvenness()
{
	int max_diff = getMaxDiff();
	double avg = (double)NNZ / subBlockNum;
	return (double)avg / (avg + max_diff);
}

// �������rateNodeArray����subsetArray��ʡ������ʱ��
void computeSubsetArray()
{
	int curBid;
	for (int i = 0; i < NNZ; ++i)
	{
		setSubBlockIdx(rateNodeArray[i]);
		curBid = rateNodeArray[i].bid;
		++subsetArray[curBid];
	}
}

// for random_shuffle
// random generator function:
ptrdiff_t myrandom(ptrdiff_t i) { return rand() % i; }
// pointer object to it:
ptrdiff_t(*p_myrandom)(ptrdiff_t) = myrandom;


// ������shuffle NNZ�汾(��������˳���rateNodeArray)
// note: Ҫ����perm�����ڸ�ԭ, permRow��СΪM, ��ʼ��Ϊ{1, 2, ..., M}
void rowShuffle(vector<int> &permRow)
{
	srand((unsigned)time(NULL));
	//cout << "rowShuffle called: " << endl;
	int userIdx;
	/*
	for (int i = 0; i < M; ++i)
	{
		permRow[i] = i;
	}
	*/
	random_shuffle(permRow.begin(), permRow.end(), p_myrandom);
#pragma omp parallel for 
	for (int i = 0; i < NNZ; ++i)
	{
		userIdx = rateNodeArray[i].u - 1;
		rateNodeArray[i].u = permRow[userIdx];	// FIXME: �������ݼ�idx�� 1~M
	}
}

// ������shuffle NNZ�汾(��������˳���rateNodeArray)
// note: Ҫ����perm�����ڸ�ԭ, permColumn��СΪN, ��ʼ��Ϊ{1, 2, ..., N}
void columnShuffle(vector<int> &permColumn)
{
	srand((unsigned)time(NULL));
	//cout << "columnShuffle called: " << endl;
	int itemIdx;
	/*
	for (int i = 0; i < N; ++i)
	{
		permColumn[i] = i;
	}
	*/
	random_shuffle(permColumn.begin(), permColumn.end(), p_myrandom);
	
	// debug:
	//printList(&permColumn[0], N);

#pragma omp parallel for 
	for (int i = 0; i < NNZ; ++i)
	{
		itemIdx = rateNodeArray[i].i - 1;
		rateNodeArray[i].i = permColumn[itemIdx];	// FIXME: �������ݼ�idx�� 1~N
	}

	// debug:
	//printNodeArrayAsMatrix();
}

// ��������任(���Ž�)
void matrixShuffle()
{
	// debug:
	//printVar("NNZ", NNZ);
	//sortRateNodeArrayBid();
	//memset(worksetArray, 0, subBlockNum*sizeof(sWorkset));
	//memset(subsetArray, 0, subBlockNum*sizeof(int));
	//setWorkset();
	// debug:
	/*
	cout << "subsetArray: ";
	printList(subsetArray, subBlockNum);
	*/

	vector<int> bestPermRow;
	vector<int> bestPermColumn;

	int min_diff = NNZ;
	int cur_diff;
	for (int i = 0; i < MAX_SHUFFLE; ++i)
	{
		// ����computeSubsetArray()�����α���
		//resetAllNode_blockIdx();	// �任������bid

		//sortRateNodeArrayBid();
		//memset(worksetArray, 0, subBlockNum*sizeof(sWorkset));
		memset(subsetArray, 0, subBlockNum*sizeof(int));
		//setWorkset();
		computeSubsetArray();

		// debug:
		/*
		cout << "subsetArray: ";
		printList(subsetArray, subBlockNum);
		*/

		cur_diff = getMaxDiff();
		if (cur_diff < min_diff)
		{
			min_diff = cur_diff;
			bestPermRow.assign(permRow.begin(), permRow.end());
			bestPermColumn.assign(permColumn.begin(), permColumn.end());
			printVar("cur_diff", cur_diff);
		}

		// ������ǰ��best�ٱ任
		rowShuffle(permRow);
		columnShuffle(permColumn);
	}
	permRow.assign(bestPermRow.begin(), bestPermRow.end());
	permColumn.assign(bestPermColumn.begin(), bestPermColumn.end());
	resetAllNode_blockIdx_label();
}

// DELETE: ��������ֱ�ӱ任Ҳ����
// ������shuffle NNZ�汾(����rateNodeArray��ԭʼ���룬����ǰ��������userIdx����)
/*
void rowShuffle()
{
cout << "rowShuffle called: " << endl;

vector<int> permVec(M);
for (int i = 0; i < M; ++i)
{
permVec[i] = i;
}
random_shuffle(permVec.begin(), permVec.end());

for (int i = 0; i < NNZ; ++i)
{
int userIdx = rateNodeArray[i].u;
while ((i < NNZ) && (userIdx == rateNodeArray[i].u))
{
rateNodeArray[i].u = permVec[i];
++i;
}
--i;	// for��++i
}
}
*/

// DELETE: ����shuffleԭʼ�� (������ɾ���ĺ�����������)
#if 0
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
#endif
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
int computeSubBlockID(int subBlockNumL, int subBlockIdxX, int subBlockIdxY)
{
	return subBlockIdxX * subBlockNumL + subBlockIdxY;
}

// ����bid��Ӧ���ӿ�x,y�±� (0��ʼ)
void getBlockXY(int bid, int &x, int &y)
{
	x = (bid / subBlockNumL);
	y = bid % subBlockNumL;
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
	*(matrixPattern + s*subBlockNumL + t) = computeSubBlockID(subBlockNumL, x, y);
}

// ��¼�ӿ�b_xy��ID: bid����ά����pattern(s, t) 
// �ӿ�bid�ǵ�s��ģʽ�еĵ�t���ӿ�, ���bid����pattern(s,t)
void setPattern(int s, int t, int bid)
{
	*(matrixPattern + s*subBlockNumL + t) = bid;
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

// debug:
// ���user item
void printNodeArrayMatrix()
{
	sortRateNodeArrayUserItem();
	map<int, vector<int>> imap;
	int userIdx;
	int itemIdx;
	for (int idx = 0; idx < NNZ; ++idx)
	{
		userIdx = rateNodeArray[idx].u;
		itemIdx = rateNodeArray[idx].i;
		imap[userIdx].push_back(itemIdx);
	}

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < imap[i].size(); ++j)
		{
			cout << imap[i][j] << "\t";
		}
		cout << endl;
	}
}






// debug:
void unitTest()
{
	CALL_FUN_TIME(initParameter())
	CALL_FUN_TIME(readFile(inputFile))
	
	/*
	sortRateNodeArrayBid();
	for (int i = 0; i < rateNodeVector.size(); ++i)
	{
		printRateNode(rateNodeArray[i]);
		printLine();
	}
	*/

	//initAllData();
	CALL_FUN_TIME(initAllData())

	//matrixShuffle();
	CALL_FUN_TIME(matrixShuffle())

	//sortRateNodeArrayBid();
	CALL_FUN_TIME(sortRateNodeArrayBid())

	// ����任��������
	memset(worksetArray, 0, subBlockNum*sizeof(sWorkset));
	memset(subsetArray, 0, subBlockNum*sizeof(int));
	CALL_FUN_TIME(setWorkset())		//setWorkset();
	CALL_FUN_TIME(setWorkseg())		//setWorkseg();
	CALL_FUN_TIME(setAllPattern())	//setAllPattern();

	fclose(stdin);
	freopen("CON", "r", stdin);   //"CON"�������̨

	// debug:
	printMatrixPattern();
	printWorksetArray();
	printMatrixWorkseg();

	///*
	CALL_FUN_TIME(solveByGPU(rateNodeArray, matrixUser, matrixItem, worksetArray,
			   mWorkseg, matrixPattern, subBlockNumL, subBlockLen, 
			   lambda, gamma, NNZ))
	//*/

	//debug:
	/*
	typeRate *matrixPredict = new typeRate[M*N];
	memset(matrixPredict, 0, M*N*sizeof(typeRate));
	matrixMultiply(matrixUser, matrixItem, matrixPredict);
	
	printMatrix(matrixUser, M, K);
	printMatrix(matrixItem, N, K);
	printMatrix(matrixPredict, M, N);
	printNodeArrayMatrix();
	*/
}


// ��ִ�к���
void execute()
{
	// ���Ի������롢�ڴ����(array matrix����ʱmemsetΪ0)
	initParameter();
	readFile(inputFile); // ���־���
	initAllData();

	
	// CPU����
	matrixShuffle();
	sortRateNodeArrayBid();
	// ����任��������
	memset(worksetArray, 0, subBlockNum*sizeof(sWorkset));
	memset(subsetArray, 0, subBlockNum*sizeof(int));
	setWorkset();
	setWorkseg();
	setAllPattern();
	

	// ����GPU����


	// ��������: ����permRow, permCol��ԭmatrixUser, matrixItem


	// ������ (model)

	
	// ���ִ�з�ʽΪ��Ԥ��
	// Ԥ�Ⲣ������

	
	// �ڴ�free
}

// sgd CPU update
void sgd_CPU()
{
    for(int iter = 0; iter < MAX_ITER; ++iter)
    {
        for(int i = 0; i < NNZ; ++i)
        {
            int userIdx = rateNodeArray[i].u - 1;
            int itemIdx = rateNodeArray[i].i - 1;
            double rate = rateNodeVector[i].rate;
            double predict = innerProduct(matrixUser, matrixItem, userIdx, itemIdx);
            double err = rate - predict;

            for(int k = 0; k < K; ++k)
            {
                double tmp = (*(matrixUser + userIdx * K + k));
                (*(matrixUser + userIdx * K + k)) += (gamma * (err * (*(matrixItem + itemIdx * K + k)) - lambda * tmp));
                (*(matrixItem + itemIdx * K + k)) += (gamma * (err * tmp - lambda * (*(matrixItem + itemIdx * K + k))));
            }
        }

        double err_sum = 0;

        for(int i = 0; i < NNZ; ++i)
        {
            int userIdx = rateNodeArray[i].u - 1;
            int itemIdx = rateNodeArray[i].i - 1;
            double rate = rateNodeArray[i].rate;
            double predict = innerProduct(matrixUser, matrixItem, userIdx, itemIdx);
            double err = rate - predict;
            err_sum += pow(err, 2);
        }

        double rmse = sqrt(err_sum / NNZ);
        cout << "iter: " << iter << "\t rmse: " << rmse << endl;
    }
}

// FGMF CPU�汾
void FGMF_CPU()
{
    CALL_FUN_TIME(initParameter())
    CALL_FUN_TIME(readFile(inputFile))
    /*
    sortRateNodeArrayBid();
    for (int i = 0; i < rateNodeVector.size(); ++i)
    {
    printRateNode(rateNodeArray[i]);
    printLine();
    }
    */
    //initAllData();
    CALL_FUN_TIME(initAllData())
    //matrixShuffle();
    //CALL_FUN_TIME(matrixShuffle())
    //sortRateNodeArrayBid();
    CALL_FUN_TIME(sortRateNodeArrayBid())
    // ����任��������
    memset(worksetArray, 0, subBlockNum * sizeof(sWorkset));
    memset(subsetArray, 0, subBlockNum * sizeof(int));
    CALL_FUN_TIME(setWorkset())     //setWorkset();
    CALL_FUN_TIME(setWorkseg())     //setWorkseg();
    CALL_FUN_TIME(setAllPattern())  //setAllPattern();

	sgd_CPU();
}