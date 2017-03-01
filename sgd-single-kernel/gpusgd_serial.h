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
using namespace std;


// TO-DO
void initParameter();


// ���־������Ԫ��
struct rateNode
{
	int u;	// userIdx
	int i;	// itemIdx
	double rate;
	int bid;
	int subBlockIdxX;
	int subBlockIdxY;
	int label;

	rateNode(){}
	rateNode(int inputU, int inputI, double inputRate) : u(inputU), i(inputI), rate(inputRate) {}
};

// ����rateNode�������ӿ��±�x y��bid
void setSubBlockIdx(rateNode &node);

// ����rateNode��label
void setLabel(rateNode &node);


// TO-DO: ��Ϊ�����ݽṹ��
// block�ӷ���
struct subBlock
{
public:
	subBlock(int blockId, int size, rateNode *nodeArray = NULL);
	~subBlock();
	void setBid();
	int getBid();
	void setSubBlockIdx();
	void getSubBlockIdx(int &x, int &y);
	void setRateNum(int size) { this->rateNum = size; }
	int getRateNum() { return this->rateNum; }
	
	// ���ӿ�������rateNode����label
	void labelNodeInSubBlock();
	// �����ӿ�bid������label��seg(bid, label): �ӿ�bid�б��б�ǩlabel������ֵ���������浽labelNumArray
		void computeSeg();
	// ����workseg(bid, label)��from��to�����ӿ�bid������ֵlabel����ʼλ��
	void computeWorkSeg();


	int bid;
	int subBlockIdxX;
	int subBlockIdxY;
	int rateNum;
	rateNode *subBlockNodeArray;
	int *labelNumArray;
	workseg *worksegArray;
};

// TO-DO: ���ɵ�subBlock���棨����workseg��
// workset: ��¼ÿ���ӿ�b_xy��bid�������־���R�ı߽�beg, end
struct workset{
	int beg;
	int end;

	// ���õĳ�Ա����
	/*
	workset() : beg(0), end(0){}
	workset(int a, int b) : beg(a), end(b) {}
	workset &operator=(const workset &rhs)
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


struct workseg {
	int from;
	int to;
};

// ���ü�¼������matrixSubset�ó�ÿ��workset��beg��end
void setWorkset(int bid);

// �����ӿ�bid�����б�ǩ������ֵ��Ŀ�����浽seg����ĵ�bid��
void computeSeg(int bid);

// ���ü�¼������seg�ó�ÿ��workseg��from��to
void setWorkseg(int bid, int tag);







// ����m�ڴ����
void newMatrix(double **m, int rowNum, int colNum);

// ����m�ڴ�����
void deleteMatrix(double **m, int rowNum, int colNum);

// �����ʼ������
void randomInitMatrix(double **m, int rowNum, int colNum);


// ������shuffle
void rowShuffle(double **a, int rowNum, int colNum);

// �������/�� shuffle (����rowShuffle)
void randomShuffleMatrix(double **m, int rowNum, int columnNum);

// ����ֿ飬��ͳ��ÿ�������Ԫ�ظ���
int blockMatrix(double **a, int rowNums, int blockLen);

// test shuffle
void randomGenerateMatrix();



// ����ӿ�bid�Ƿ�Խ��
// ����ֵ��
// 0 ��Խ��
// 1 ��ȫԽ��
// 2 ����Խ��
int checkSubBlockBoundary(int bid);

// �����ӿ�b_xy��ID�������ӿ��СΪ: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockLen, int x, int y);

// ��¼�ӿ�b_xy��ID: bid����ά����pattern(s, t) 
// �ӿ�b_xy �ǵ�s��ģʽ�еĵ�t���ӿ�, ���computeSubBlockID(z, x, y)�Ľ������pattern(s,t)
void setPattern(int **matrixPattern, int s, int t, int subBlockLen, int x, int y);

// ���أ�subset(x, y)
// �ӿ� b_xy ����������ֵ����(����Ԫ��)
int computeSubset(double **m, int subBlockIdxX, int subBlockIdxY, int subBlockLen);

// ����bid��Ӧ���ӿ�x,y�±�
void getBlockXY(int bid, int &x, int &y);

// ����Rui�������ӿ�x,y�±�
void getBlockXY(int u, int i, int &x, int &y);

#endif