# pragma once

#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <time.h>  
#include <stdio.h>
#include <windows.h> 

#include"NLM.h"
#include"AKNN.h"

using namespace std;
using namespace cv;

class VideoDenoisingME{
	string videoName;
	vector<Mat>frames;
	int width;
	int height;
	int H;
	int S;
	int K;

public:
	VideoDenoisingME();
	int processing(vector<Mat>&dstFrames, string videoName, string storeName, int K, int H, int lenPatch);
	void videoDenoising(vector<Mat>framesSrc, vector<Mat>&framesOut);
private:
	void singalChannelHandle(vector<Mat>&dstFrames);
	void multiChannelHandle(vector<Mat>&dstFrames);
};

extern VideoDenoisingME mVideoDenoisingME;