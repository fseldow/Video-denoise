# pragma once

//#include "cv.h"
//#include "highgui.h"
#include"mex\OpticalFlow.h"
#include <iostream>
#include <math.h>
#include <windows.h> 
#include"AKNN.h"

using namespace std;
//using namespace cv;

#ifndef PI
#define PI 3.1415926
#endif

struct NLM :cv::ParallelLoopBody{
	float gama;
	float sigma_p;
	int H;
	int K;
	int S;
	vector<cv::Mat> &frames;
	cv::Mat &dst;
public:
	cv::Rect board;
	void operator() (const cv::Range& range) const;
	NLM(int H, int K, int lenPatch, vector<cv::Mat>&frames, cv::Mat &dst);
	double getSigma_t(cv::Mat src_t, cv::Mat src_f, KNN z, DImage vx, DImage vy)const;
	double NLM_Estimate(cv::Point3i z, double _sigma_t, vector<ImgKNN> NNF)const;
	double weightedSSD(cv::Point3i p, cv::Point3i q)const;
private:
	
	
	
	//void setSigma_t(double sigma_t)const;
	
};