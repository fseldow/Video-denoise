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


class AKNN{
private:
	Mat img;                         //img in certain flame
	Mat imgSrc;                      //img with original patch
	Mat patch;                       //source patch
	Mat patchSurroungding[4];        //
	int K;                           //K nearest neighbors
	int S;                           //half edge of searching patch
	double sigma;
	int nImgRows;
	int nImgCols;
	Point2i pPatch;
	vector<NeighborPatch> neighbors;
public:
	AKNN(const Mat img, const Mat imgSrc);
	vector<NeighborPatch> getV(Point2i pPatch,int K,int lenPatch);

private:
	void initation();
	void progagation(int iter,int k);
	void randomSearch(int i);
	void handleQueue(NeighborPatch mNeighborPatch);
	void operation();
	double calculateDistance(Point2i p, Mat patch);
	double calculateDistance(Mat q, Mat p);
	Point2d generateNormal2dVector();

}; 