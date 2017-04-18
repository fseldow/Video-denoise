# pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include"AKNN.h"

using namespace std;
using namespace cv;

#define PI 3.1415926

class NLM{
	float gama;
	float sigma_t;
	float sigma_p;
	int H;
	int K;
	int S;
	vector<Mat>frames;
public:
	NLM(int H, int K, int lenPatch, vector<Mat>frames);
	double NLM_Estimate(Point3i z);
private:
	double weightedSSD(Point3i p, Point3i q);
};