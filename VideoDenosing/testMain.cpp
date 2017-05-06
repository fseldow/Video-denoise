#include <iostream>
#include "VideoDenoisingME.h"
#include<windows.h>
#include"mex\OpticalFlow.h"
#include <time.h>

using namespace std;
//using namespace cv;




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
	
	


	double start = GetTickCount();
	vector<cv::Mat> test;
	cv::Mat result;
	cv::VideoCapture capture("E:\\C++\\video1.avi");
	cv::Mat frame;
	/*capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;
	capture >> frame;*/
	double fps = capture.get(CV_CAP_PROP_FPS);
	//获得原始视频的高度和宽度
	//cv::Size size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	///创建一个视频文件参数分别表示  新建视频的名称 视频压缩的编码格式 新建视频的帧率 新建视频的图像大小
	//cv::VideoWriter writer("E:\\C++\\poor_result.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
	//while(1){
	for (int i = 0; i < 50;i++){
		cv::Mat frame;
		capture >> frame;
		if (frame.empty())break;
		test.push_back(frame);
		//writer.write(frame);
	}
	cout << "read complete" << endl;

	


	vector<cv::Mat> vResult;
	VideoDenoisingME b;
	b.processing(test, vResult, 11, 11, 7);
	for (int i = 0; i < vResult.size(); i++){
		//writer .write( vResult[i]);
	}
	double end = GetTickCount();
	cout << "Total " << vResult.size() << " use time " << end - start << endl;
	int aaaa = 0;

	//test AKNN
	//while (1){
	//	double start = GetTickCount();
	//	cv::Mat srcimg, tarimg;
	//	srcimg = cv::imread("frame_large1.jpg");
	//	//srcimg.convertTo(srcimg, CV_64FC1);

	//	tarimg = cv::imread("frame_large2.jpg");
	//	//tarimg.convertTo(tarimg, CV_64FC1);

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