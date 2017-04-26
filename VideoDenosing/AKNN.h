# pragma once

//#include "cv.h"
//#include "highgui.h"
#include"mex\OpticalFlow.h"
#include <math.h>
#include <iostream>
#include <windows.h> 
#include<ctime>

using namespace std;
//using namespace cv;

# define ALPHA   0.5
# define NUMMOD  1000



struct NeighborPatch{
	cv::Point2i p;
	double distance;
	NeighborPatch(cv::Point2i mP, double mDistance){
		p = mP; distance = mDistance;
	}
};

typedef vector<NeighborPatch> KNN;
typedef KNN ** ImgKNN;

class AKNN{
private:
	cv::Mat imgDst;                         //img in certain flame
	const cv::Mat &imgSrc;                      //img with original patch

	int K;                           //K nearest neighbors
	int S;                           //half edge of searching patch

	double sigma;

	int nSrcRows;
	int nSrcCols;
	int nDstRows;
	int nDstCols;

	cv::Point2i **NNF;
	double **offset;
	ImgKNN &KNNF;

public:
	AKNN(const cv::Mat &src, ImgKNN &_knnf);
	~AKNN();
	int getV(int K, int lenPatch);
	void setDst(cv::Mat &dst);

private:
	void initation();
	void progagation(cv::Point2i patch, int odd);
	void randomSearch(cv::Point2i pSrc, cv::Point2i nnf);
	void handleQueue(KNN &knn, NeighborPatch);
	void operation();
	double calculateDistance(cv::Point2i pDst, cv::Point2i pSrc);
	double calculateDistance(cv::Mat q, cv::Mat p);
	cv::Point2d generateNormal2dVector();
	int getMinIndex(double a, double b, double c);
}; 