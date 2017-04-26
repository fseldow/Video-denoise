#include <iostream>
#include "VideoDenoisingME.h"
#include<windows.h>
#include"mex\OpticalFlow.h"
#include <time.h>

using namespace std;
//using namespace cv;


void test()
{
	int a = 0;
	for (int i = 0; i<100000000; i++)
		a++;
}

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
	//cv::Mat src;
	//src = cv::imread("E:\\C++\\PatchMatch-master\\lena.bmp", 0);
	//src.convertTo(src, CV_64FC1);
	//DImage dst;
	//mat2DImage(src, dst);
	//const double *test1DImage = dst.data();
	//cout << test1DImage[2] << endl;
	//DImage aaa;
	//aaa.imread("E:\\C++\\PatchMatch-master\\lena.bmp");
	//const double *testDImage = aaa.data();
	//cout << testDImage[2] << endl;
	//max(2,3);
	//cv::Vector<cv::Mat>aaaa;
	//cv::waitKey();

	//ImgKNN a;
	//a = new KNN *[1000];
	//KNN aa;
	//for (int i = 0; i < 121; i++){
	//	aa.push_back(NeighborPatch(Point2i(-1, -1), 1));
	//}
	////for (int i = 0; i < 11; i++){
	//	//a[i] = new KNN *[1000];
	//	for (int j = 0; j < 1000; j++){
	//		a[j] = new KNN[1000];
	//		for (int k = 0; k < 1000; k++){
	//			a[j][k] = aa;
	//		}
	//	}
	////}


	/*clock_t t1 = clock();
    #pragma omp parallel for  
	for (int i = 0; i<8; i++)
		test();
	clock_t t2 = clock();
	std::cout << "time: " << t2 - t1 << std::endl;*/



	vector<cv::Mat>frames;
	VideoDenoisingME b;
	b.processing(frames,"E:\\C++\\video1.mp4", "aa", 11, 5, 7);


	//vector<Mat> test;
	//Mat result;
	//VideoCapture capture("E:\\C++\\video1.mp4");
	//for (int i = 0; i < 11;i++){
	//	Mat frame;
	//	capture >> frame;
	//	if (frame.empty())break;
	//	test.push_back(frame);
	//	//imshow("video", frame);
	//	//waitKey();
	//}


	//double start = GetTickCount();
	//fastNlMeansDenoisingColoredMulti(test, result, 5, 11,36);
	//double end = GetTickCount();
	//cout << end - start << endl;
	//imwrite("result.png",result);
	//imshow("aa", result);
	//waitKey();
	//for (int i = 0; i < test.size(); i++){
	//	imshow("video", test[i]);
	//	waitKey();
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