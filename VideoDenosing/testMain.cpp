#include <iostream>
#include "VideoDenoisingME.h"
#include<windows.h>
#include"mex\OpticalFlow.h"
#include <time.h>

using namespace std;
//using namespace cv;



// use for test Optical FLow
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

int main(){
	double start = GetTickCount();
	vector<cv::Mat> test;
	cv::Mat result;
	cv::VideoCapture capture("E:\\source.avi");

	double fps = capture.get(CV_CAP_PROP_FPS);
	cv::Size size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	cv::VideoWriter writer("E:\\C++\\result.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, size);


	while(1){
		cv::Mat frame;
		capture >> frame;
		if (frame.empty())break;
		test.push_back(frame);
	}
	cout << "read complete" << endl;

	
	vector<cv::Mat> vResult;
	VideoDenoisingME ::processing(test, vResult, 11, 11, 7);
	for (int i = 0; i < vResult.size(); i++){
		writer .write( vResult[i]);
	}
	double end = GetTickCount();
	cout << "Total " << vResult.size() << " use time " << end - start << endl;
	


	////test Optical Flow
	//cv::Mat frame1, frame2;
	//frame1 = cv::imread("frame_large1.jpg");
	//frame2 = cv::imread("frame_large2.jpg");
	//DImage pre, cur, warp,vx,vy;
	//mat2DImage(frame1, pre);
	//mat2DImage(frame2, cur);
	//double alpha = 0.012;
	//double ratio = 0.75;
	//int minWidth = 20;
	//int nOuterFPIterations = 7;
	//int nInnerFPIterations = 1;
	//int nCGIterations = 30;
	//OpticalFlow::Coarse2FineFlow(vx, vy, warp, pre, cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nCGIterations);

	//test AKNN
	//while (1){
	//	double start = GetTickCount();
	//	cv::Mat srcimg, tarimg;
	//	srcimg = cv::imread("frame_large1.jpg");

	//	tarimg = cv::imread("frame_large2.jpg");

	//	//OpticalFlow::Coarse2FineFlow(srcimg);
	//	//cv::Rect certain(100, 100, 200, 200);
	//	//cv::Mat tt = srcimg(certain);
	//	ImgKNN result;
	//	AKNN<cv::Vec3b> maknn(srcimg, result);
	//	maknn.setDst(tarimg);
	//	maknn.getV(11, 7);
	//	double end = GetTickCount();
	//	cout <<"use "<< end - start << endl;
	//	KNN m_knn = result[25][25];
	//	int a = 0;
	//}
	return 0;
}