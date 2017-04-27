#include <iostream>
#include "VideoDenoisingME.h"
#include<windows.h>
#include"mex\OpticalFlow.h"
#include <time.h>

using namespace std;
//using namespace cv;




void mat2DImage(cv::Mat srcMat, DImage& dstDImage){
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
			dstDImage.pData[i*width + j] = srcMat.at<double>(i, j);
		}
	}
}

int main(){
	vector<cv::Mat>frames;
	VideoDenoisingME b;
	b.processing(frames,"E:\\C++\\video1.mp4", "aa", 11, 5, 7);


	//vector<cv::Mat> test;
	//cv::Mat result;
	//cv::VideoCapture capture("E:\\C++\\video1_poor.mp4");
	//while(1){
	//	cv::Mat frame;
	//	capture >> frame;
	//	if (frame.empty())break;
	//	test.push_back(frame);
	//}
	//VideoDenoisingME b;
	//vector<cv::Mat> src;
	//for (int i = 20; i < 30; i++){
	//	src.push_back()
	//}
	

	//while (1){
	//	double start = GetTickCount();
	//	Mat srcimg, tarimg;
	//	srcimg = imread("E:\\C++\\PatchMatch-master\\lena.bmp", 0);
	//	//srcimg.convertTo(srcimg, CV_64FC1);

	//	tarimg = imread("E:\\C++\\PatchMatch-master\\barbara.bmp", 0);
	//	//tarimg.convertTo(tarimg, CV_64FC1);

	//	//OpticalFlow::Coarse2FineFlow(srcimg);
	//	Rect certain(100, 100, 200, 200);
	//	Mat tt = srcimg(certain);
	//	ImgKNN result;
	//	AKNN maknn(tt, result);
	//	maknn.setDst(tt);
	//	maknn.getV(11, 7);
	//	double end = GetTickCount();
	//	cout <<"use "<< end - start << endl;
	//	KNN m_knn = result[25][25];
	//	int a = 0;
	//}
	return 0;
}