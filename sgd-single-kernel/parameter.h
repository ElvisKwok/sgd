#ifndef __PARAMETER_H_
#define __PARAMETER_H_

extern double **matrixRate;	// 评分矩阵
extern double **matrixUser;
extern double **matrixItem;
extern int M;	// matrixRate 行数
extern int N;	// matrixRate 列数
extern int K;	// 隐含向量维数
extern int subBlockNumL;	// subBlockNumL * subBlockNumL个子块
extern int subBlockLen;	// 子块大小为 subBlockLen * subBlockLen

#endif