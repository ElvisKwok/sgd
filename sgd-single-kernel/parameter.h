#ifndef __PARAMETER_H_
#define __PARAMETER_H_


typedef double typeRate;

extern int MAX_ITER;			// ����������
extern double lambda;			// ����ϵ��
extern double gamma;			// ѧϰ��


extern typeRate **matrixRate;	// ���־��� size: M * N
extern typeRate *matrixUser;	// size: K * M
extern typeRate *matrixItem;	// size: K * N
extern int M;	// matrixRate ����
extern int N;	// matrixRate ����
extern int K;	// ��������ά��
extern int subBlockNumL;	// subBlockNumL * subBlockNumL���ӿ�
extern int subBlockNum;	// �ӿ�����Ŀ
extern int subBlockLen;	// �ӿ��СΪ subBlockLen * subBlockLen
extern int subBlockNodeNum; // �ӿ��Сsize������0���0��



#endif