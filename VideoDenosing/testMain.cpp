#include <opencv2/opencv.hpp>
#include <iostream>
#include "VideoDenoisingME.h"
#include <time.h>

using namespace std;
using namespace cv;


void test()
{
	int a = 0;
	for (int i = 0; i<100000000; i++)
		a++;
}

int main(){
	/*clock_t t1 = clock();
    #pragma omp parallel for  
	for (int i = 0; i<8; i++)
		test();
	clock_t t2 = clock();
	std::cout << "time: " << t2 - t1 << std::endl;*/



	vector<Mat>frames;
	VideoDenoisingME a;
	a.processing(frames,"E:\\C++\\video1.mp4", "aa", 11, 5, 7);



	/*VideoCapture capture("LXH1.mp4");
	while (1){
		Mat frame;
		capture >> frame;
		if (frame.empty())break;
		imshow("video", frame);
		waitKey();
	}*/
	//while (1){
	//	Mat srcimg, tarimg;
	//	srcimg = imread("E:\\C++\\PatchMatch-master\\lena.bmp");
	//	srcimg.convertTo(srcimg, CV_64FC1);
	//	srcimg = srcimg*1.0 / 255;
	//	tarimg = imread("E:\\C++\\PatchMatch-master\\barbara.bmp");
	//	tarimg.convertTo(tarimg, CV_64FC1);
	//	tarimg = tarimg*1.0 / 255;
	//	Rect certain(253, 253, 7, 7);
	//	AKNN maknn(tarimg, srcimg);
	//	vector<NeighborPatch> testvector = maknn.getV(Point2i(256, 256), 11, 7);
	//}
	return 0;
}