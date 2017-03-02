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
using namespace std;


// TO-DO
void initParameter();


// 评分矩阵非零元素
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

// 计算rateNode所属的子块下标x y和bid
void setSubBlockIdx(sRateNode &node);

// 计算rateNode的label
void setLabel(sRateNode &node);


// block子方块
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

// 根据bid设置子块x y坐标
void setSubBlockIdx(sSubBlock &subBlock);

// 设置子块包含的非0元素个数
void setRateNum(sSubBlock &subBlock);

// 取消：直接复制指针，不要重复分配空间
/*
// 分配子块的node数组空间
void allocSubBlockNodeArray(sSubBlock &subBlock);
*/

// 对子块内所有rateNode进行label
void labelNodeInSubBlock(sSubBlock &subBlock);

// 计算子块bid中所有标签的评价值数目，保存到seg数组的第bid行
void computeSeg(int bid);

// 利用记录的数组seg得出每个workseg的from和to
// 计算workseg(bid, label)的from和to，即子块bid中评价值label的起始位置
void computeWorkseg(int bid, int tag);

// 设置子块所属的pattern
void setSubBlockPattern(sSubBlock &subBlock);

// 计算所有子块的pattern到二维数组pattern
void setAllPattern();

// sort比较谓语pred
bool compare_label(sRateNode a, sRateNode b);

// 根据label对子块内所有rateNode进行排序
void sortLabelInSubBlock(sSubBlock &subBlock);

// 调用sortLabelInSubBlock，对所有子块进行label排序
void sortLabelAll();


// 取消：直接复制指针，不要重复分配空间
/*
// 释放子块的动态内存空间
void destroySubBlock(sSubBlock &subBlock);
*/

// sWorkset: 记录每个子块b_xy（bid）在评分矩阵R的边界beg, end
struct sWorkset{
	int beg;
	int end;

	// 无用的成员函数
	/*
	sWorkset() : beg(0), end(0){}
	sWorkset(int a, int b) : beg(a), end(b) {}
	sWorkset &operator=(const sWorkset &rhs)
	{
	// 首先检测等号右边的是否就是左边的对象本，若是本对象本身,则直接返回
	if (this == &rhs)
	{
	return *this;
	}

	// 复制等号右边的成员到左边的对象中
	this->beg = rhs.beg;
	this->end = rhs.end;

	// 把等号左边的对象再次传出
	// 目的是为了支持连等 eg:    a=b=c 系统首先运行 b=c
	// 然后运行 a= ( b=c的返回值,这里应该是复制c值后的b对象)
	return *this;
	}
	void print(){ cout << beg << "\t" << end << endl; }
	void setBeg(int subBlockIdxX, int subBlockIdxY);
	void setEnd(int subBlockIdxX, int subBlockIdxY);

	void setFromInput(int a, int b) { beg = a; end = b; }
	*/
};

// 利用记录的数组matrixSubset得出每个workset的beg和end
void setWorkset(int bid);


struct sWorkseg {
	int from;
	int to;
};



// 矩阵m内存分配
void newMatrix(typeRate **m, int rowNum, int colNum);

// 矩阵m内存销毁
void deleteMatrix(typeRate **m, int rowNum, int colNum);

// 随机初始化矩阵
void randomInitMatrix(typeRate **m, int rowNum, int colNum);


// 矩阵行shuffle
void rowShuffle(typeRate **a, int rowNum, int colNum);

// 矩阵的行/列 shuffle (调用rowShuffle)
void randomShuffleMatrix(typeRate **m, int rowNum, int columnNum);

// 矩阵分块，并统计每块包含的元素个数
int blockMatrix(typeRate **a, int rowNums, int blockLen);

// test shuffle
void randomGenerateMatrix();



// 检查子块bid是否越界
// 返回值：
// 0 不越界
// 1 完全越界
// 2 部分越界
int checkSubBlockBoundary(int bid);

// 计算子块b_xy的ID，其中子块大小为: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockLen, int x, int y);

// 计算bid对应的子块x,y下标
void getBlockXY(int bid, int &x, int &y);

// 计算Rui所属的子块x,y下标
void getBlockXY(int u, int i, int &x, int &y);


// 记录子块b_xy的ID: bid到二维数组pattern(s, t) 
// 子块b_xy 是第s种模式中的第t个子块, 则把computeSubBlockID(z, x, y)的结果放入pattern(s,t)
void setPattern(int s, int t, int x, int y);

// 记录子块b_xy的ID: bid到二维数组pattern(s, t) 
// 子块bid是第s种模式中的第t个子块, 则把bid放入pattern(s,t)
void setPattern(int s, int t, int bid);

// 返回：subset(x, y)
// 子块 b_xy 包含的评价值个数(非零元素)
int computeSubset(int subBlockIdxX, int subBlockIdxY);

// 记录所有subset(x, y)到全局二维数组matrixSubset
// 子块 b_xy 包含的评价值个数(非零元素)
void computeAllSubset();

#endif