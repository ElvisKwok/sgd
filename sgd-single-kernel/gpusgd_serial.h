#ifndef __GPUSGD_SERIAL_H_
#define __GPUSGD_SERIAL_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>


#include "basic_func.h"
#include "parameter.h"
#include "parallelReadFile.h"
using namespace std;

// debug
template <typename T>
void printVar(string varName, const T &var)
{
	cout << varName << ": \t" << var << endl;
}
inline void printLine() { cout << "#######################################" << endl; }
void unitTest();

// ��ȡ�����ļ���
// M, N
// (user, item, rate)
void readFile(string fileName = "input.txt");

// ���ѵ��ģ��
void writeFile(string fileName = "model.txt");

// ���Ԥ����
// resultArray��user_item�����˳��
void writeFile(typeRate *result, int predictNum, string fileName = "predict_result.txt");

// ��ʼ������
void initParameter();

// ���������M, N ���Ի�subBlockά�Ȳ���
void initBlockDimension();

// sWorkset: ��¼ÿ���ӿ�b_xy��bid�������־���R�ı߽�beg, end
struct sWorkset{
	int beg;
	int end;

	// ���õĳ�Ա����
	/*
	sWorkset() : beg(0), end(0){}
	sWorkset(int a, int b) : beg(a), end(b) {}
	sWorkset &operator=(const sWorkset &rhs)
	{
	// ���ȼ��Ⱥ��ұߵ��Ƿ������ߵĶ��󱾣����Ǳ�������,��ֱ�ӷ���
	if (this == &rhs)
	{
	return *this;
	}

	// ���ƵȺ��ұߵĳ�Ա����ߵĶ�����
	this->beg = rhs.beg;
	this->end = rhs.end;

	// �ѵȺ���ߵĶ����ٴδ���
	// Ŀ����Ϊ��֧������ eg:    a=b=c ϵͳ�������� b=c
	// Ȼ������ a= ( b=c�ķ���ֵ,����Ӧ���Ǹ���cֵ���b����)
	return *this;
	}
	void print(){ cout << beg << "\t" << end << endl; }
	void setBeg(int subBlockIdxX, int subBlockIdxY);
	void setEnd(int subBlockIdxX, int subBlockIdxY);

	void setFromInput(int a, int b) { beg = a; end = b; }
	*/
};


// sWorkseg: ��¼�ӿ�bid��ÿ������ֵ��tag�ı߽�from��to
struct sWorkseg {
	int from;
	int to;
};


// ���־������Ԫ��
struct sRateNode
{
	int u;	// userIdx
	int i;	// itemIdx
	typeRate rate;
	int bid;
	int subBlockIdxX;
	int subBlockIdxY;
	int label;

	sRateNode(){}
	sRateNode(int inputU, int inputI, typeRate inputRate) :
		u(inputU), i(inputI), rate(inputRate) {}
};

// ����rateNode�������ӿ��±�x y��bid
void setSubBlockIdx(sRateNode &node);

// ����rateNode��label
void setLabel(sRateNode &node);

// sort�Ƚ�ν��pred: �Ȱ�bid����bid��ͬ��label����
bool compare_bid_label(sRateNode a, sRateNode b);

// ����bid��rateNodeArray��������
void sortRateNodeArrayBid();

// block�ӷ���
struct sSubBlock
{
	int bid;
	int subBlockIdxX;
	int subBlockIdxY;
	int rateNum;
	sRateNode *subBlockNodeArray;
	int pattern;

	sSubBlock() {}
	sSubBlock(int blockId, int size, sRateNode *nodeArray = NULL): 
		bid(blockId), rateNum(size), subBlockNodeArray(nodeArray){}
};

// DELETE
// setSubBlockIdx()
// setRateNum() 
// allocSubBlockNodeArray() 
// labelNodeInSubBlock()
/*
// ����bid�����ӿ�x y����
void setSubBlockIdx(sSubBlock &subBlock);

// �����ӿ�����ķ�0Ԫ�ظ���
void setRateNum(sSubBlock &subBlock);


// ȡ����ֱ�Ӹ���ָ�룬��Ҫ�ظ�����ռ�
// DELETE
// �����ӿ��node����ռ�
//void allocSubBlockNodeArray(sSubBlock &subBlock);


// ���ӿ�������rateNode����label
void labelNodeInSubBlock(sSubBlock &subBlock);
*/




// DELETE
// workseg����
/*
// �����ӿ�bid�����б�ǩ������ֵ��Ŀ�����浽seg����ĵ�bid��
void computeSeg(int bid);

// ���ü�¼������seg�ó�ÿ��workseg��from��to
// ����workseg(bid, label)��from��to�����ӿ�bid������ֵlabel����ʼλ��
void computeWorkseg(int bid, int tag);
*/

// ���������ӿ��pattern����ά����pattern
void setAllPattern();

// DELETE
// ����pattern
/*
// �����ӿ�������pattern
void setSubBlockPattern(sSubBlock &subBlock);

// ���������ӿ��pattern����ά����pattern
void setAllPattern();
*/


// DELETE
// �ӿ���rateNode����
/*
// sort�Ƚ�ν��pred
bool compare_label(sRateNode a, sRateNode b);

// ����label���ӿ�������rateNode��������
void sortLabelInSubBlock(sSubBlock &subBlock);

// ����sortLabelInSubBlock���������ӿ����label����
void sortLabelAll();
*/

// ȡ����ֱ�Ӹ���ָ�룬��Ҫ�ظ�����ռ�
// DELETE
/*
// �ͷ��ӿ�Ķ�̬�ڴ�ռ�
void destroySubBlock(sSubBlock &subBlock);
*/

// DELETE
// ���ü�¼������matrixSubset�ó�����workset��beg��end
/*
void setWorkset(int bid);
*/

// ɨ��bid������rateNodeArray, �ó�ÿ��workset��beg��end�����浽worksetArray
// ͬʱͳ��ÿ��subset��ֵ,���浽subsetArray
void setWorkset();

// ɨ��bid_label������rateNodeArray, �ó�ÿ��workseg��from��to�����浽mWorkseg
void setWorkseg();


// TO-DO
// ����array�ڴ����
template <typename T>
void newArray(T *array, int n);

// ����m�ڴ����
void newMatrix(typeRate **m, int rowNum, int colNum);

// ����m�ڴ�����
void deleteMatrix(typeRate **m, int rowNum, int colNum);

// �����ʼ������
void randomInitMatrix(typeRate **m, int rowNum, int colNum);


// ������shuffle
void rowShuffle(typeRate **a, int rowNum, int colNum);

// �������/�� shuffle (����rowShuffle)
void randomShuffleMatrix(typeRate **m, int rowNum, int columnNum);

// ����ֿ飬��ͳ��ÿ�������Ԫ�ظ���
int blockMatrix(typeRate **a, int rowNums, int blockLen);

// test shuffle
void randomGenerateMatrix();



// DELETE
// ����ӿ�bid�Ƿ�Խ��
/*
// ����ֵ��
// 0 ��Խ��
// 1 ��ȫԽ��
// 2 ����Խ��
int checkSubBlockBoundary(int bid);
*/

// �����ӿ�b_xy��ID�������ӿ��СΪ: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockLen, int x, int y);

// ����bid��Ӧ���ӿ�x,y�±�
void getBlockXY(int bid, int &x, int &y);

// ����Rui�������ӿ�x,y�±�
void getBlockXY(int u, int i, int &x, int &y);


// ��¼�ӿ�b_xy��ID: bid����ά����pattern(s, t) 
// �ӿ�b_xy �ǵ�s��ģʽ�еĵ�t���ӿ�, ���computeSubBlockID(z, x, y)�Ľ������pattern(s,t)
void setPattern(int s, int t, int x, int y);

// ��¼�ӿ�b_xy��ID: bid����ά����pattern(s, t) 
// �ӿ�bid�ǵ�s��ģʽ�еĵ�t���ӿ�, ���bid����pattern(s,t)
void setPattern(int s, int t, int bid);

// DELETE
// ԭsubset����ļ���
/*
// ���أ�subset(x, y)
// �ӿ� b_xy ����������ֵ����(����Ԫ��)
int computeSubset(int subBlockIdxX, int subBlockIdxY);

// ��¼����subset(x, y)��ȫ�ֶ�ά����matrixSubset
// �ӿ� b_xy ����������ֵ����(����Ԫ��)
void computeAllSubset();
*/




// ��ִ�к���
void execute();

#endif