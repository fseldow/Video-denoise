#include"VideoDenoisingME.h"

VideoDenoisingME::VideoDenoisingME(){

}

int VideoDenoisingME::processing(vector<Mat>&dstFrames, string videoName,string storeName,int K,int H,int lenPatch){
	this->K = K;
	this->H = H;
	this->S = (lenPatch - 1) / 2;
	this->videoName = videoName;
	Mat frame;
	VideoCapture capture(videoName);
	Mat tempFrame;
	capture >> tempFrame;
	if (tempFrame.empty())return -1;
	width = tempFrame.cols;
	height = tempFrame.rows;

	if (tempFrame.channels() == 1){
		singalChannelHandle(dstFrames);
	}
	else{
		multiChannelHandle(dstFrames);
	}
	return 0;
}

void VideoDenoisingME::singalChannelHandle(vector<Mat>&dstFrames){
	vector<Mat>gFrames;

	VideoCapture capture(videoName);
	cout << "read file" << endl;
	for (int i = 0; i < H * 2 + 1; i++){
		Mat tempFrame;
		capture >> tempFrame;
		if (tempFrame.empty())break;
		if (tempFrame.type() != CV_64FC1){
			tempFrame.convertTo(tempFrame, CV_64FC1);
			//tempFrame = tempFrame * 1.0 / 255;
		}
		gFrames.push_back(tempFrame);
	}
	while (1){
		
		Mat tempFrame;
		capture >> tempFrame;
		if (tempFrame.empty())break;

		if (tempFrame.type() != CV_64FC1){
			tempFrame.convertTo(tempFrame, CV_64FC1);
			//tempFrame = tempFrame * 1.0 / 255;
		}

		gFrames._Pop_back_n(0);
		gFrames.push_back(tempFrame);
	}

	cout << "denoising..." << endl;
	videoDenoising(gFrames, dstFrames);
}

void VideoDenoisingME::multiChannelHandle(vector<Mat>&dstFrames){
	vector<Mat>rFrames, bFrames, gFrames;
	VideoCapture capture(videoName);
	cout << "read file" << endl;

	for (int i = 0; i < H * 2 + 1; i++){
		Mat tempFrame;
		capture >> tempFrame;
		if (tempFrame.empty())break;

		if (tempFrame.type() != CV_64FC1){
			tempFrame.convertTo(tempFrame, CV_64FC1);
			tempFrame = tempFrame * 1.0 / 255;
		}

		vector<Mat>singleFrameRBG;
		split(tempFrame, singleFrameRBG);
		rFrames.push_back(singleFrameRBG[0]);
		bFrames.push_back(singleFrameRBG[1]);
		gFrames.push_back(singleFrameRBG[2]);
	}
	int n=0;
	while (1){
		vector<Mat>rOutFrames, bOutFrames, gOutFrames;
		//cout <<"frame: "<< n++ << endl;
		Mat tempFrame;
		capture >> tempFrame;
		imwrite("ori.jpg", tempFrame);

		double start, end;
		start = GetTickCount();
		Mat dst;
		fastNlMeansDenoisingColored(tempFrame(Rect(500, 50, 150, 150)), dst, 3, 3, 11, 21);
		end = GetTickCount();
		cout << end - start << endl;

		imshow("ff", tempFrame);
		imshow("df", dst);
		waitKey();

		if (tempFrame.empty())break;
		imshow("ori", tempFrame(Rect(500, 50, 150, 150)));
		if (tempFrame.type() != CV_64FC1){
			tempFrame.convertTo(tempFrame, CV_64FC1);
			tempFrame = tempFrame * 1.0 / 255;
		}

		start = GetTickCount();

		vector<Mat>singleFrameRBG;
		split(tempFrame,singleFrameRBG);
		rFrames.erase(rFrames.begin());
		rFrames.push_back(singleFrameRBG[0]);
		videoDenoising(rFrames, rOutFrames);



		bFrames.erase(bFrames.begin());
		bFrames.push_back(singleFrameRBG[1]);
		videoDenoising(bFrames, bOutFrames);



		gFrames.erase(gFrames.begin());
		gFrames.push_back(singleFrameRBG[2]);
		videoDenoising(gFrames, gOutFrames);




		vector<Mat>singleOutFrameRBG;
		singleOutFrameRBG.push_back(rOutFrames[H]);
		singleOutFrameRBG.push_back(bOutFrames[H]);
		singleOutFrameRBG.push_back(gOutFrames[H]);
		merge(singleFrameRBG, tempFrame);

		end = GetTickCount();
		cout << end - start << endl;
		imwrite("out.jpg", tempFrame);
		imshow("frame", tempFrame(Rect(500, 50, 150, 150)));
		waitKey();
		dstFrames.push_back(tempFrame);

	}

	//cout << "denoising..." << endl;
	//vector<Mat>rOutFrames, bOutFrames, gOutFrames;
	//videoDenoising(rFrames, rOutFrames);
	//videoDenoising(bFrames, bOutFrames);
	//videoDenoising(gFrames, gOutFrames);
	//for (int i = 0; i < rFrames.size(); i++)
	//{
	//	//ºÏ²¢
	//	vector<Mat>singleFrameRBG;
	//	singleFrameRBG.push_back(rOutFrames[i]);
	//	singleFrameRBG.push_back(bOutFrames[i]);
	//	singleFrameRBG.push_back(gOutFrames[i]);
	//	merge(singleFrameRBG, dstFrames[i]);
	//}
}

void VideoDenoisingME::videoDenoising(vector<Mat>framesSrc, vector<Mat>&framesOut){
	//×ªdouble

	/*cout << "change type..." << endl;
	if (framesSrc[0].type() != CV_64FC1){
		for (int i = 0; i < framesSrc.size(); i++){
			framesSrc[i].convertTo(framesSrc[i], CV_64FC1);
			framesSrc[i] = framesSrc[i] * 1.0 / 255;
		}
	}*/
	NLM mNLM(H, K, 2 * S + 1, framesSrc);

	cout << "copy..." << endl;
	if (framesOut.empty()){
		//¿½±´
		for (int f = 0; f < framesSrc.size() ; f++){
			Mat temp;
			framesSrc[f].copyTo(temp);
			framesOut.push_back(temp);
		}
	}

	cout << "start NLM..." << endl;
	double start = GetTickCount();
	for (int f = H; f < framesSrc.size() - H; f++){
		for (int i = 500; i < 650; i++){
			//cout << i << endl;
			for (int j = 50; j < 200; j++){
				framesOut[f].at<double>(j, i) = mNLM.NLM_Estimate(Point3i(i, j, f));
				//double a=framesSrc[f].at<double>(j, i);
				//double b = framesOut[f].at<double>(j, i);
				//cout << b - a << endl;
				//cout << framesOut[f].at<double>(i, j) << " " << framesSrc[f].at<double>(i, j) << endl;
				//cout << "loc: " << H << " " << i << " " << j << endl;
			}
		}
	}
	double end = GetTickCount();
	cout << end - start << endl;
	waitKey();
	cout << "endNLM" << endl;
}