# pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include"AKNN.h"

using namespace std;
using namespace cv;

#define PI 3.1415926

struct NLM :ParallelLoopBody{
	float gama;
	float sigma_p;
	int H;
	int K;
	int S;
	vector<Mat>frames;
	vector<Mat> &dst;
public:
	void operator() (const Range& range) const;
	NLM(int H, int K, int lenPatch, vector<Mat>frames, vector<Mat> &dst);
	double getSigma_t(Mat src_t, Mat src_f)const;
	double NLM_Estimate(Point3i z, double _sigma_t)const;
	
private:
	double weightedSSD(Point3i p, Point3i q)const;
	
	
	//void setSigma_t(double sigma_t)const;
	
};