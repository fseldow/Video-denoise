#include <opencv2/opencv.hpp>
#include <iostream>
#include "VideoDenoisingME.h"

using namespace std;
using namespace cv;

int main(){
	vector<Mat>frames;
	VideoDenoisingME a;
	a.processing(frames,"lxh1.mp4", "aa", 11, 5, 7);

	/*VideoCapture capture("LXH1.mp4");
	while (1){
		Mat frame;
		capture >> frame;
		if (frame.empty())break;
		imshow("video", frame);
		waitKey();
	}*/
	///*mat srcimg, tarimg;
	//srcimg = imread("lena.bmp");
	//srcimg.convertto(srcimg, cv_64fc1);
	//srcimg = srcimg*1.0 / 255;
	//tarimg = imread("barbara.bmp");
	//tarimg.convertto(tarimg, cv_64fc1);
	//tarimg = tarimg*1.0 / 255;
	//rect certain(253,253,7,7);
	//aknn maknn(tarimg,srcimg);
	//vector<neighborpatch> testvector = maknn.getv(point2i(256,256), 11, 7);*/
	return 0;
}