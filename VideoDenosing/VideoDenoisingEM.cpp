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

		if (tempFrame.type() != CV_64FC1){
			tempFrame.convertTo(tempFrame, CV_64FC1);
			//tempFrame = tempFrame * 1.0 / 255;
		}

		

		vector<Mat>singleFrameRBG;
		split(tempFrame,singleFrameRBG);

		rFrames.erase(rFrames.begin());
		rFrames.push_back(singleFrameRBG[0]);

		imshow("rSrc", rFrames[H](Rect(700,50,100,100))*1.0/255);
		//cout << rFrames[H](Rect(700, 50, 100, 100)) << endl;
		//waitKey();
		videoDenoising(rFrames, rOutFrames);
		imshow("rOut", rOutFrames[H](Rect(700, 50, 100, 100)) * 1.0 / 255);
		waitKey();
		//parallel_for_(Range(50, 100), NLM(H, K, 2 * S + 1, rFrames, rOutFrames));
		

		bFrames.erase(bFrames.begin( ));
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






	cout << "start NLM..." << endl;
	double start = GetTickCount();
	for (int f = H; f < framesSrc.size() - H; f++){
		////////////////////////////////////////////////////////////
		//get all neighbors patches
		////////////////////////////////////////////////////////////
		vector<ImgKNN> nearestNeighbors(framesSrc.size());
//#pragma omp parallel for 

		for (int n_frame = f - H; n_frame <= f + H; n_frame++){
			cout << "f: " << n_frame << endl;
			double start = GetTickCount();
			ImgKNN result;
			AKNN mAKNN(framesSrc[n_frame], framesSrc[f], result);
			mAKNN.getV(K, 2 * S + 1);
			nearestNeighbors[n_frame] = result;
			double end = GetTickCount();
			cout << "AKNN use time : " << end - start << endl;
		}


		double end = GetTickCount();
		cout <<"Total AKNN use time : "<< end - start << endl;
		double sigma_t = mNLM.getSigma_t(framesSrc[f], framesSrc[f + 1]);
#pragma omp parallel for 
		for (int x = 700; x < 800; x++){
			for (int y = 50; y < 150; y++){
				framesOut[f].at<double>(y, x) = mNLM.NLM_Estimate(Point3i(x, y, f), sigma_t, nearestNeighbors);
			}
		}
		nearestNeighbors.clear();
	}
	start = GetTickCount();
	double end = GetTickCount();
	cout << end - start << endl;
	//waitKey();
	cout << "endNLM" << endl;
}