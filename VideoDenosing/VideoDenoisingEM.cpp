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
	parallel_for_(Range(0, width), NLM(H, K, 2 * S + 1, gFrames, dstFrames));
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
			//tempFrame = tempFrame * 1.0 / 255;
		}

		vector<Mat>singleFrameRBG;
		split(tempFrame, singleFrameRBG);
		rFrames.push_back(singleFrameRBG[0]);
		bFrames.push_back(singleFrameRBG[1]);
		gFrames.push_back(singleFrameRBG[2]);
	}
	

	while (1){
		vector<Mat>rOutFrames, bOutFrames, gOutFrames;
		//cout <<"frame: "<< n++ << endl;
		Mat tempFrame;
		capture >> tempFrame;
		imwrite("ori.jpg", tempFrame);

		double start, end;
		//start = GetTickCount();
		//Mat dst;
		//fastNlMeansDenoisingColored(tempFrame(Rect(300, 50, 100, 50)), dst, 3, 3, 11, 21);
		//end = GetTickCount();
		//cout << end - start << endl;

		//imshow("ff", tempFrame);
		//imshow("df", dst);
		//waitKey();

		if (tempFrame.empty())break;
		imshow("ori", tempFrame);
		if (tempFrame.type() != CV_64FC1){
			tempFrame.convertTo(tempFrame, CV_64FC1);
			//tempFrame = tempFrame * 1.0 / 255;
		}

		

		vector<Mat>singleFrameRBG;
		split(tempFrame,singleFrameRBG);

		rFrames.erase(rFrames.begin());
		rFrames.push_back(singleFrameRBG[0]);
		//////////////////////////////////////////////////////////////
		start = GetTickCount();
		videoDenoising(rFrames, rOutFrames);
		//parallel_for_(Range(50, 100), NLM(H, K, 2 * S + 1, rFrames, rOutFrames));
		end = GetTickCount();
		cout << end - start << endl;
		//////////////////////////////////////////////////////////////

		bFrames.erase(bFrames.begin());
		bFrames.push_back(singleFrameRBG[1]);
		videoDenoising(bFrames, bOutFrames);
		//parallel_for_(Range(50,100), NLM(H, K, 2 * S + 1, bFrames, bOutFrames));



		gFrames.erase(gFrames.begin());
		gFrames.push_back(singleFrameRBG[2]);
		videoDenoising(gFrames, gOutFrames);
		//parallel_for_(Range(50, 100), NLM(H, K, 2 * S + 1, gFrames, gOutFrames));




		vector<Mat>singleOutFrameRBG;
		singleOutFrameRBG.push_back(rOutFrames[H]);
		singleOutFrameRBG.push_back(bOutFrames[H]);
		singleOutFrameRBG.push_back(gOutFrames[H]);
		merge(singleOutFrameRBG, tempFrame);
		
		
		imwrite("out.jpg", tempFrame);
		tempFrame = tempFrame*1.0 / 255.0;
		imshow("frame", tempFrame);
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
	
	NLM mNLM(H, K, 2 * S + 1, framesSrc, framesOut);

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
		double sigma_t=mNLM.getSigma_t(framesOut[f], framesOut[f+1]);
		//mNLM.setSigma_t(sigma_t);
//#pragma omp parallel for 
		for (int x = 100; x < 200; x++){
			//cout << x << endl;
//#pragma omp parallel for
			for (int y = 50; y < 150; y++){
				framesOut[f].at<double>(y, x) = mNLM.NLM_Estimate(Point3i(x, y, f),sigma_t);
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
	//waitKey();
	cout << "endNLM" << endl;
}