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
#include"AKNN.hpp"
#include"NLM.hpp"

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
	int processing(vector<cv::Mat>srcFrames, vector<cv::Mat>&dstFrames, int K, int temporalWindowSize, int patchWindowSize);
	void videoDenoising(vector<cv::Mat>framesSrc, cv::Mat&framesOut,vector<DImage>vx,vector<DImage>vy, int _K, int temporalWindowSize, int patchWindowSize);

	//template<class T>

private:
	void mat2DImage(cv::Mat srcMat, DImage &dstDImage){
		int width = srcMat.cols;
		int height = srcMat.rows;
		if (dstDImage.height() == 0 && dstDImage.width() == 0){
			dstDImage.allocate(width, height, srcMat.channels());
		}
		else{
			dstDImage.imresize(width, height);
		}
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				int nPixel = i*width + j;
				if (srcMat.channels()>1){
					for (int c = 0; c < srcMat.channels(); c++){
						dstDImage.pData[nPixel*srcMat.channels() + c] = srcMat.at<cv::Vec3b>(i, j)[c]/255.0;
					}
				}
				else{
					dstDImage.pData[i*width + j] = srcMat.at<uchar>(i, j)/255.0;
				}
			}
		}
	}
};

