# pragma once

//#include "cv.h"
//#include "highgui.h"
#include"mex\OpticalFlow.h"
#include <math.h>
#include <iostream>
#include <time.h>  
#include <stdio.h>
#include <windows.h> 

//#include"NLM.h"
#include"AKNN.h"

using namespace std;
//using namespace cv;

#define GAMA 0.9

#ifndef PI
#define PI 3.1415926
#endif

class VideoDenoisingME{
	

	string videoName;
	vector<cv::Mat>frames;
	int width;
	int height;
	int H;
	int S;
	int K;

	float gama;
	float sigma_p;

public:
	VideoDenoisingME();
	int processing(vector<cv::Mat>&dstFrames, string videoName, string storeName, int K, int H, int lenPatch);
	void videoDenoising(vector<cv::Mat>framesSrc, cv::Mat&framesOut, int _K, int temporalWindowSize, int searchWindowSize);
private:
	void singalChannelHandle(vector<cv::Mat>&dstFrames);
	void multiChannelHandle(vector<cv::Mat>&dstFrames);
	double getSigma_t(cv::Mat src_t, cv::Mat src_f, DImage vx, DImage vy);
	double weightedSSD(cv::Point3i p, cv::Point3i q, vector<cv::Mat>frames);

	void mat2DImage(cv::Mat srcMat, DImage &dstDImage){
		int width = srcMat.cols;
		int height = srcMat.rows;
		if (dstDImage.height() == 0 && dstDImage.width() == 0){
			dstDImage.allocate(width, height);
		}
		else{
			dstDImage.imresize(width, height);
		}
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				dstDImage.pData[i*width+j] = srcMat.at<uchar>(i, j);
			}
		}
	}
};

extern VideoDenoisingME mVideoDenoisingME;