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

		//imshow("rSrc", rFrames[H](Rect(700,50,100,100))*1.0/255);
		//cout << rFrames[H](Rect(700, 50, 100, 100)) << endl;
		//waitKey();
		videoDenoising(rFrames, rOutFrames);
		//imshow("rOut", rOutFrames[H](Rect(700, 50, 100, 100)) * 1.0 / 255);
		//waitKey();
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
	
	if (framesSrc.size() != 2 * H + 1)return;
	ImgKNN result;
	NLM mNLM(H, K, 2 * S + 1, framesSrc, framesOut);
	AKNN mAKNN(framesSrc[H], result);
	
	double sigma_t = mNLM.getSigma_t(framesSrc[H], framesSrc[H + 1]);

	Mat Z(width, height, CV_64FC1, Scalar(0));
	Mat I(width, height, CV_64FC1, Scalar(0));

	////////////////////////////////////////////////////////////
	//get all neighbors patches
	////////////////////////////////////////////////////////////

	for (int nFrame = 0; nFrame < 2 * H + 1; nFrame++){
			mAKNN.setDst(framesSrc[H]);
			double startAKNN = GetTickCount();
			mAKNN.getV(K, 2 * S + 1);
			double endAKNN = GetTickCount();
			KNN matchPatch = result[750][150];
			cout << "AKNN completed : " << endAKNN - startAKNN << endl;
		
		//////////////////////////////////////////////////////////////////////////////////////
		//start calculate every pixel via NLM
		/////////////////////////////////////////////////////////////////////////////////////
		double startNLM=GetTickCount();
 		for (int x = 700; x < 850; x++){
			for (int y = 50; y < 200; y++){
				KNN matchPatch = result[x][y];
				for (int k = 0; k < K; k++){
					
					Point3i p = Point3i(x, y, H);
					double Dw = mNLM.weightedSSD(Point3i(matchPatch[k].p.x, matchPatch[k].p.y, nFrame), p);
					double temp = pow(0.9, abs(nFrame - p.z)) * exp(-(Dw / (2 * sigma_t*sigma_t)));
					Z.at<double>(y,x) += temp;
					I.at<double>(y, x) += framesSrc[nFrame].at<double>(matchPatch[k].p) * temp;
				}
			}
		}
		double endNLM = GetTickCount();
		cout << "NLM completed : " << endNLM - startNLM << endl;
	}
	I = I / Z;
	I(Rect(700, 50, 150, 150)).copyTo(framesOut[H](Rect(700, 50, 150, 150)));
	//imshow("test", I(Rect(700,50,100,100))*1.0 / 255);
	//waitKey();
	
//	AKNN mAKNN(NULL, framesSrc[f], result);
//	cout << "start NLM..." << endl;
//	double start = GetTickCount();
//
//
//	////////////////////////////////////////////////////////////
//	//get all neighbors patches
//	////////////////////////////////////////////////////////////
//
//	vector<ImgKNN> nearestNeighbors(framesSrc.size());
//	for (int n_frame = 0; n_frame <= H*2+1; n_frame++){
//		cout << "f: " << n_frame << endl;
//		double start = GetTickCount();
//		ImgKNN result;
//		AKNN mAKNN(framesSrc[n_frame], framesSrc[f], result);
//		mAKNN.getV(K, 2 * S + 1);
//		nearestNeighbors[n_frame] = result;
//		double end = GetTickCount();
//		cout << "AKNN use time : " << end - start << endl;
//	}
//
//
//	double end = GetTickCount();
//	cout <<"Total AKNN use time : "<< end - start << endl;
//	double sigma_t = mNLM.getSigma_t(framesSrc[H], framesSrc[H + 1]);
//
//
//	//////////////////////////////////////////////////////////////////////////////////////
//	//start calculate every pixel via NLM
//	/////////////////////////////////////////////////////////////////////////////////////
//	NLM mNLM(H, K, 2 * S + 1, framesSrc, framesOut);
//
//#pragma omp parallel for 
//	for (int x = 700; x < 800; x++){
//		for (int y = 50; y < 150; y++){
//			framesOut[H].at<double>(y, x) = mNLM.NLM_Estimate(Point3i(x, y, f), sigma_t, nearestNeighbors);
//		}
//	}
//	nearestNeighbors.clear();
//
//	start = GetTickCount();
//	double end = GetTickCount();
//	cout << end - start << endl;
//	//waitKey();
//	cout << "endNLM" << endl;
}