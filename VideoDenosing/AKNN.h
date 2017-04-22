# pragma once

#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include<ctime>

using namespace std;
using namespace cv;

# define ALPHA   0.5
# define NUMMOD  1000



struct NeighborPatch{
	Point2i p;
	double distance;
	NeighborPatch(Point2i mP, double mDistance){
		p = mP; distance = mDistance;
	}
};

typedef vector<NeighborPatch> KNN;
typedef KNN ** ImgKNN;

class AKNN{
private:
	const Mat &imgDst;                         //img in certain flame
	const Mat &imgSrc;                      //img with original patch

	int K;                           //K nearest neighbors
	int S;                           //half edge of searching patch

	double sigma;

	int nSrcRows;
	int nSrcCols;
	int nDstRows;
	int nDstCols;

	Point2i **NNF;
	double **offset;
	ImgKNN &KNNF;

public:
	AKNN(const Mat &dst, const Mat &src, ImgKNN &_knnf);
	~AKNN();
	int getV(int K, int lenPatch);

private:
	void initation();
	void progagation(Point2i patch, int odd);
	void randomSearch(Point2i pSrc, Point2i nnf);
	void handleQueue(KNN &knn, NeighborPatch);
	void operation();
	double calculateDistance(Point2i pDst, Point2i pSrc);
	double calculateDistance(Mat q, Mat p);
	Point2d generateNormal2dVector();
	int getMinIndex(double a, double b, double c);
}; 