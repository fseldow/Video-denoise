#include"VideoDenoisingME.h"

VideoDenoisingME::VideoDenoisingME(){

}

int VideoDenoisingME::processing(vector<cv::Mat>&dstFrames, string videoName, string storeName, int K, int H, int lenPatch){
	this->K = K;
	this->H = H;
	this->S = (lenPatch - 1) / 2;
	this->videoName = videoName;
	cv::Mat frame;
	cv::VideoCapture capture(videoName);
	cv::Mat tempFrame;
	capture >> tempFrame;
	if (tempFrame.empty())return -1;
	width = tempFrame.cols;
	height = tempFrame.rows;

	sigma_p = S / 2.0;
	gama = GAMA;

	if (tempFrame.channels() == 1){
		singalChannelHandle(dstFrames);
	}
	else{
		multiChannelHandle(dstFrames);
	}
	return 0;
}

void VideoDenoisingME::singalChannelHandle(vector<cv::Mat>&dstFrames){
	vector<cv::Mat>gFrames;

	cv::VideoCapture capture(videoName);
	cout << "read file" << endl;
	
	for (int i = 0; i < H * 2 + 1; i++){
		cv::Mat tempFrame;
		capture >> tempFrame;
		if (tempFrame.empty())break;
		//if (tempFrame.type() != CV_64FC1){
		//	tempFrame.convertTo(tempFrame, CV_64FC1);
		//	//tempFrame = tempFrame * 1.0 / 255;
		//}
		gFrames.push_back(tempFrame);
	}
	while (1){
		
		cv::Mat tempFrame;
		capture >> tempFrame;
		if (tempFrame.empty())break;

		//if (tempFrame.type() != CV_64FC1){
		//	tempFrame.convertTo(tempFrame, CV_64FC1);
		//	//tempFrame = tempFrame * 1.0 / 255;
		//}

		gFrames._Pop_back_n(0);
		gFrames.push_back(tempFrame);
	}

	cout << "denoising..." << endl;
	//parallel_for_(cv::Range(0, width), NLM(H, K, 2 * S + 1, gFrames, dstFrames));
}

void VideoDenoisingME::multiChannelHandle(vector<cv::Mat>&dstFrames){
	vector<cv::Mat>rFrames, bFrames, gFrames;
	cv::VideoCapture capture(videoName);
	cv::Mat tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;

	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	capture >> tempFrame;
	cout << "read file" << endl;

	for (int i = 0; i < H * 2 + 1; i++){
		cv::Mat tempFrame;
		capture >> tempFrame;
		if (tempFrame.empty())break;

		//if (tempFrame.type() != CV_64FC1){
		//	tempFrame.convertTo(tempFrame, CV_64FC1);
		//	//tempFrame = tempFrame * 1.0 / 255;
		//}

		vector<cv::Mat>singleFrameRBG;
		split(tempFrame, singleFrameRBG);
		rFrames.push_back(singleFrameRBG[0]);
		bFrames.push_back(singleFrameRBG[1]);
		gFrames.push_back(singleFrameRBG[2]);
	}
	

	while (1){
		cv::Mat rOutFrames, bOutFrames, gOutFrames;
		//cout <<"frame: "<< n++ << endl;
		cv::Mat tempFrame;
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

		//if (tempFrame.type() != CV_64FC1){
		//	tempFrame.convertTo(tempFrame, CV_64FC1);
		//	//tempFrame = tempFrame * 1.0 / 255;
		//}

		

		double startTime = GetTickCount();

		vector<cv::Mat>singleFrameRBG;
		split(tempFrame,singleFrameRBG);

		rFrames.erase(rFrames.begin());
		rFrames.push_back(singleFrameRBG[0]);

		//imshow("rSrc", rFrames[H](Rect(700,50,100,100))*1.0/255);
		//cout << rFrames[H](Rect(700, 50, 100, 100)) << endl;
		//waitKey();
		videoDenoising(rFrames, rOutFrames,K,2*H+1,2*S+1);
		//imshow("rOut", rOutFrames[H](Rect(700, 50, 100, 100)) * 1.0 / 255);
		//waitKey();
		//parallel_for_(Range(50, 100), NLM(H, K, 2 * S + 1, rFrames, rOutFrames));
		

		bFrames.erase(bFrames.begin( ));
		bFrames.push_back(singleFrameRBG[1]);
		videoDenoising(bFrames, bOutFrames, K, 2 * H + 1, 2 * S + 1);
		//parallel_for_(Range(50,100), NLM(H, K, 2 * S + 1, bFrames, bOutFrames));



		gFrames.erase(gFrames.begin());
		gFrames.push_back(singleFrameRBG[2]);
		videoDenoising(gFrames, gOutFrames, K, 2 * H + 1, 2 * S + 1);
		//parallel_for_(Range(50, 100), NLM(H, K, 2 * S + 1, gFrames, gOutFrames));




		vector<cv::Mat>singleOutFrameRBG;
		singleOutFrameRBG.push_back(rOutFrames);
		singleOutFrameRBG.push_back(bOutFrames);
		singleOutFrameRBG.push_back(gOutFrames);
		merge(singleOutFrameRBG, tempFrame);
		
		double endTime=GetTickCount();
		cout << "Total Time: " << endTime - startTime << endl;

		imwrite("out.jpg", tempFrame);
		//tempFrame = tempFrame*1.0 / 255.0;
		imshow("frame", tempFrame);
		cv::waitKey();
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
//-----------------------------------------------------------------------------------
//function to denoise the structured noise
//-----------------------------------------------------------------------------------
void VideoDenoisingME::videoDenoising(vector<cv::Mat>framesSrc, cv::Mat&framesOut, int _K, int temporalWindowSize, int searchWindowSize){
	this->K = _K;
	this->H = temporalWindowSize / 2;
	temporalWindowSize = H * 2 + 1;
	this->S = searchWindowSize / 2;
	searchWindowSize = S * 2 + 1;
	
	if (framesSrc.size() != 2 * H + 1)return;

	
	//NLM mNLM(H, K, 2 * S + 1, framesSrc, framesOut);

	if (framesOut.empty()){
		//¿½±´
		//#pragma omp parallel for
		framesSrc[H].copyTo(framesOut);
	}

	//-------------------------------------------------------------------------------
	//get all neighbors patches
	//-------------------------------------------------------------------------------

	cout << "------------------------------------------------\nstart AKNN" << endl;
	double startAKNN = GetTickCount();

	ImgKNN result;
	AKNN mAKNN(framesSrc[H], result);
	mAKNN.setDst(framesSrc[H]);
	mAKNN.getV(K, 2 * S + 1);
	KNN mknn = result[200][60];
	double endAKNN = GetTickCount();
	cout << "use time : " << endAKNN - startAKNN << endl;

	//DImage pre, cur,flow;
	//mat2DImage(framesSrc[H+1], pre);
	//mat2DImage(framesSrc[H+2], cur);
	//OpticalFlow::ComputeOpticalFlow(pre, cur, flow);
	//int pixel = 60 * width + 200;
	//int xxxx = flow.data()[pixel * 2];
	//int yyyy = flow.data()[pixel * 2 + 1];

	//----------------------------------------------------------------------------------------------------
	//Optical Flow
	//----------------------------------------------------------------------------------------------------

	cout << "------------------------------------------------\nstart Optical Flow" << endl;
	double startOF = GetTickCount();

	vector<DImage>v_vx, v_vy;
	DImage pre, cur;
	mat2DImage(framesSrc[H], pre);


	for (int f = 0; f < 2 * H + 1; f++){
		DImage vx, vy, warp;
		if (f == H){
			vx.allocate(width, height);
			vy.allocate(width, height);
			for (int i = 0; i < height; i++){
				for (int j = 0; j < width; j++){
					vx.pData[i*width + j] = 0;
					vy.pData[i*width + j] = 0;
				}
			}
			v_vx.push_back(vx);
			v_vy.push_back(vy);
			continue;
		}
		mat2DImage(framesSrc[f], cur);
		OpticalFlow::Coarse2FineFlow(vx, vy, warp, pre, cur, 1, 0.7, 30, 3, 1, 40);
		int pixel =  60 * width + 200;
		cout << vx. data()[pixel] <<'\t'<< vy.data()[pixel] << endl;
		v_vx.push_back(vx);
		v_vy.push_back(vy);
	}
	pre.clear();
	cur.clear();

	double endOF = GetTickCount();
	cout << "use time : " << endOF - startOF << endl;

	//-------------------------------------------------------------------------------
	//calculate sigma_t
	//-------------------------------------------------------------------------------

	double sigma_t = getSigma_t(framesSrc[H], framesSrc[H + 1],v_vx[H+1],v_vy[H+1]);




	//-------------------------------------------------------------------------------
	//start calculate every pixel via NLM
	//-------------------------------------------------------------------------------

	cout << "------------------------------------------------\nstart NLM" << endl;
	double startNLM = GetTickCount();

	int x_start=S+2, x_end=width-S-3;
	int y_start = S + 2, y_end = height - S - 3;

#pragma omp parallel for
	for (int x = x_start; x < x_end; x++){
		for (int y = y_start; y < y_end; y++){
			double I = 0;
			double Z = 0;
			KNN matchPatchCurrentFrame = result[x][y];
			for (int f = 0; f < H*2+1; f++){
				for (int k = 0; k < K; k++){
					cv::Point3i p = cv::Point3i(x, y, H);
					cv::Point3i neighbor = cv::Point3i(
						result[x][y][k].p.x + v_vx[f].pData[y*width + x],
						result[x][y][k].p.y + v_vy[f].pData[y*width + x],
						f);
					if (neighbor.x>S && neighbor.x<width -S-1 && neighbor.y>S && neighbor.y<height -S- 1){
						double Dw = weightedSSD(neighbor, p, framesSrc);
						double temp = pow(GAMA, abs(f - H)) * exp(-(Dw / (2 * sigma_t*sigma_t)));
						Z += temp;
						I += framesSrc[f].at<uchar>(neighbor.y, neighbor.x) * temp;
					}
				}
			}
			I /= Z;
			framesOut.at<uchar>(y, x) = cv::saturate_cast<uchar> (I);
		}
	}

	double endNLM = GetTickCount();
	cout << "use time : " << endNLM - startNLM << endl;
}



double VideoDenoisingME::getSigma_t(cv::Mat src_t, cv::Mat src_f,  DImage vx, DImage vy){
	double sigma_n = 20;
	int rows = src_t.rows;
	int cols = src_t.cols;
	double J;
	double sigma_temp1, sigma_temp2, preSigma_n = 0;
	cv::Mat alfa(cv::Size(cols, rows), CV_64FC1);

	while (abs(sigma_n - preSigma_n)>0.1){
		//while (1){
		preSigma_n = sigma_n;
		sigma_temp1 = 0;
		sigma_temp2 = 0;

		for (int x = 0; x < cols; x++){
		    for (int y = 0; y < rows; y++){
				cv::Point2i z(x, y);
				cv::Point2i pNeighbor(x + vx.pData[y*width + x], y + vy.pData[y*width + x]);
				if (pNeighbor.x < 0 || pNeighbor.y < 0 || pNeighbor.x >= cols || pNeighbor.y >= rows)continue;
				J = src_t.at<uchar>(z) -src_f.at<uchar>(pNeighbor);
				alfa.at<double>(z) = exp(-J / (2 * sigma_n*sigma_n)) / (exp(-J / (2 * sigma_n*sigma_n)) + 0.5*sigma_n*pow(2 * PI, 0.5));
				//int test = alfa.at<double>(i, j);
				sigma_temp1 += J*J*alfa.at<double>(z);
				sigma_temp2 += alfa.at<double>(z);
		    }
		}
		sigma_n = pow(sigma_temp1 / sigma_temp2, 0.5);
	}
	return sigma_n;
}

double VideoDenoisingME::weightedSSD(cv::Point3i p, cv::Point3i q,vector<cv::Mat>_frames){
	double D = 0;
	double D_norm = 0;
	for (int i = -S; i <= S; i++){
		for (int j = -S; j <= S; j++){
			double temp = exp(-(i*i + j*j) / (2.0 * sigma_p*sigma_p));
			D += pow(_frames[p.z].at<uchar>(p.y + j, p.x + i) - _frames[q.z].at<uchar>(p.y + j, p.x + i), 2)*temp;
			D_norm += temp;
		}
	}
	D /= D_norm;
	return D;
}