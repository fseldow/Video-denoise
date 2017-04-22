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
	ImgKNN a;
	a = new KNN *[1000];
	KNN aa;
	for (int i = 0; i < 121; i++){
		aa.push_back(NeighborPatch(Point2i(-1, -1), 1));
	}
	//for (int i = 0; i < 11; i++){
		//a[i] = new KNN *[1000];
		for (int j = 0; j < 1000; j++){
			a[j] = new KNN[1000];
			for (int k = 0; k < 1000; k++){
				a[j][k] = aa;
			}
		}
	//}


	/*clock_t t1 = clock();
    #pragma omp parallel for  
	for (int i = 0; i<8; i++)
		test();
	clock_t t2 = clock();
	std::cout << "time: " << t2 - t1 << std::endl;*/



	vector<Mat>frames;
	VideoDenoisingME b;
	b.processing(frames,"E:\\C++\\video1.mp4", "aa", 11, 5, 7);



	/*VideoCapture capture("LXH1.mp4");
	while (1){
		Mat frame;
		capture >> frame;
		if (frame.empty())break;
		imshow("video", frame);
		waitKey();
	}*/
	while (1){
		double start = GetTickCount();
		Mat srcimg, tarimg;
		srcimg = imread("E:\\C++\\PatchMatch-master\\lena.bmp");
		srcimg.convertTo(srcimg, CV_64FC1);
		srcimg = srcimg*1.0 / 255;
		tarimg = imread("E:\\C++\\PatchMatch-master\\barbara.bmp");
		tarimg.convertTo(tarimg, CV_64FC1);
		tarimg = tarimg*1.0 / 255;
		Rect certain(253, 253, 100, 100);
		ImgKNN result;
		AKNN maknn(tarimg, srcimg(certain), result);
		maknn.getV(11,7);
		double end = GetTickCount();
		cout <<"use "<< end - start << endl;
		KNN m_knn = result[25][25];
		int a = 0;
	}
	return 0;
}