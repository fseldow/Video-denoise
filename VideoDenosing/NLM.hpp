# include"AKNN.hpp"
#include"mex\OpticalFlow.h"
#include"nlmeans_denoising_commons.hpp"


const double Gama = 0.9;


template <typename T>
class NLM{
	int temporalWindowSize;
	int patchWindowSize;
	int H;
	int S;
	int K;
	double sigma_p;
	int height;
	int width;
	vector<cv::Mat>srcFrames;
	cv::Mat&framesOut;

	ImgKNN KNNF;
	vector<DImage>vxFlow;
	vector<DImage>vyFlow;

public:
	NLM(vector<cv::Mat>srcFrames, cv::Mat&framesOut, int K, int temporalWindowSize, int patchWindowSize, vector<DImage>vx,	vector<DImage>vy);
	int operation();
private:
	void estimateNoise(double *result,cv::Mat src_t, cv::Mat src_f, DImage vx, DImage vy);
	void estimateNoise2(double *result, cv::Mat src_t);
	double weightedSSD(cv::Point3i p, cv::Point3i q, vector<cv::Mat>_frames);

};

template<class T>
NLM<T>::NLM(vector<cv::Mat>_srcFrames, cv::Mat&dstFrames, int _K, int _temporalWindowSize, int _patchWindowSize, vector<DImage>_vxFlow, vector<DImage>_vyFlow)
: srcFrames(_srcFrames), framesOut(dstFrames), K(_K), vxFlow(_vxFlow), vyFlow(_vyFlow)
{
	if (framesOut.empty()){
		framesOut.create(srcFrames[0].size(), srcFrames[0].type());
	}
	
	
	this->width = srcFrames[0].cols;
	this->height = srcFrames[0].rows;
	this->H = _temporalWindowSize / 2;
	this->temporalWindowSize = H * 2 + 1;
	this->S = _patchWindowSize / 2;
	this->patchWindowSize = S * 2 + 1;

	this->sigma_p = S / 2.0;

	srcFrames[H].copyTo(framesOut);
}


template<class T >
int NLM<T>:: operation(){
	//--------------------------------------------------------------------------------
	//estimate noise
	//--------------------------------------------------------------------------------
	double sigma_t[3] = { 0 };
	double start = GetTickCount();
	estimateNoise(sigma_t, srcFrames[H], srcFrames[H + 1], vxFlow[H], vyFlow[H]);
	//estimateNoise2(sigma_t,srcFrames[H]);

	//--------------------------------------------------------------------------------
	//AKNN
	//--------------------------------------------------------------------------------
	cout << "start searching neighbor patch" << endl;
	AKNN<T> mAKNN(srcFrames[H],KNNF);
	mAKNN.setDst(srcFrames[H]);
	mAKNN.getV(K, patchWindowSize);
	double end= GetTickCount();
	cout <<"extra use time :"<< end - start << endl;
	//--------------------------------------------------------------------------------
	//NLM
	//--------------------------------------------------------------------------------
	cout << "start NLM" << endl;
	int x_start = S + 2, x_end = width - S - 3;
	int y_start = S + 2, y_end = height - S - 3;

	double NLMstart = GetTickCount();
#pragma omp parallel for
	for (int x = x_start; x < x_end; x++){
		for (int y = y_start; y < y_end; y++){
			cv::Point3i p = cv::Point3i(x, y, H);
			double I[3] = { 0 };
			double Z[3] = { 0 };
			//KNN matchPatchCurrentFrame = result[x][y];
			for (int k = 0; k < K; k++){
				//AKNN tempKNNF;
				cv::Point2i neighborInCurrentFrame = KNNF[x][y][k].p;
				cv::Point3i neighbor = cv::Point3i(
					neighborInCurrentFrame.x,
					neighborInCurrentFrame.y,
					H);
				if (neighbor.x <= S || neighbor.x >= width - S - 1 || neighbor.y <= S || neighbor.y >= height - S - 1)break;
				double Dw = weightedSSD(neighbor, p, srcFrames);
				double weight[3];
				for (int i = 0; i < sizeof(T); i++){
					weight[i] = 1;//pow(0.9, abs(f - H)) * exp(-(Dw / (2 * sigma_t[i] * sigma_t[i])));
					Z[i] += weight[i];
				}
				incWithWeight(I, weight,srcFrames[H].at<T>(neighbor.y, neighbor.x));
				
				//寻找后几帧
				for (int f = H + 1; f < temporalWindowSize ; f++){
					cv::Point2i tempNeighborInCurrentFrame = neighborInCurrentFrame;
					neighborInCurrentFrame.x += vxFlow[f - 1].pData[tempNeighborInCurrentFrame.y*width + tempNeighborInCurrentFrame.x];
					neighborInCurrentFrame.y += vyFlow[f - 1].pData[tempNeighborInCurrentFrame.y*width + tempNeighborInCurrentFrame.x];
					//neighborInCurrentFrame = tempNeighborInCurrentFrame;
					cv::Point3i neighbor = cv::Point3i(
						neighborInCurrentFrame.x,
						neighborInCurrentFrame.y,
						f);
					if (neighbor.x <= S || neighbor.x >= width - S - 1 || neighbor.y <= S || neighbor.y >= height - S - 1)break;
					double Dw = weightedSSD(neighbor, p, srcFrames);
					double weight[3];
					for (int i = 0; i < sizeof(T); i++){
						weight[i] = pow(Gama, abs(f - H)) * exp(-(Dw / (2 * sigma_t[i] * sigma_t[i])));
						Z[i] += weight[i];
					}
					incWithWeight(I, weight, srcFrames[f].at<T>(neighbor.y, neighbor.x));
					
				}
				//寻找前几帧
				neighborInCurrentFrame = KNNF[x][y][k].p;
				for (int f = H -1; f >=0; f--){
					cv::Point2i tempNeighborInCurrentFrame = neighborInCurrentFrame;
					neighborInCurrentFrame.x -= vxFlow[f].pData[tempNeighborInCurrentFrame.y*width + tempNeighborInCurrentFrame.x];
					neighborInCurrentFrame.y -= vyFlow[f].pData[tempNeighborInCurrentFrame.y*width + tempNeighborInCurrentFrame.x];
					cv::Point3i neighbor = cv::Point3i(
						neighborInCurrentFrame.x,
						neighborInCurrentFrame.y,
						f);
					if (neighbor.x <= S || neighbor.x >= width - S - 1 || neighbor.y <= S || neighbor.y >= height - S - 1)break;
					double Dw = weightedSSD(neighbor, p, srcFrames);
					double weight[3];
					for (int i = 0; i < sizeof(T); i++){
						weight[i] = pow(Gama, abs(f - H)) * exp(-(Dw / (2 * sigma_t[i] * sigma_t[i])));
						Z[i] += weight[i];
					}
					incWithWeight(I, weight, srcFrames[f].at<T>(neighbor.y, neighbor.x));
					
				}
			}
			for (int i = 0; i < sizeof(T); i++){
				I[i] /= Z[i];
			}
			framesOut.at<T>(y, x) = saturateCastFromArray<T>(I);
		}
	}
	double NLMend = GetTickCount();
	cout << "NLM use time:" << NLMend - NLMstart << endl;
	return 0;
}

template<class T>
double NLM<T>::weightedSSD(cv::Point3i p, cv::Point3i q, vector<cv::Mat>_frames){
	double D = 0;
	double D_norm = 0;
	for (int i = -S; i <= S; i++){
		for (int j = -S; j <= S; j++){
			double temp = exp(-(i*i + j*j) / (2.0 * sigma_p*sigma_p));
			D +=calcDistance(_frames[p.z].at<T>(p.y + j, p.x + i) ,_frames[q.z].at<T>(p.y + j, p.x + i))*temp;
			D_norm += temp;
		}
	}
	D /= D_norm;
	return D;
}

template<class T>
void NLM<T>::estimateNoise(double *sigma_t,cv::Mat src_t, cv::Mat src_f, DImage vx, DImage vy){
	//double sigma_t[3];
	for (int i = 0; i < sizeof(T); i++){
		double sigma_n = 1;
		double J;
		double sigma_temp1, sigma_temp2, preSigma_n = 0;
		cv::Mat alfa(cv::Size(width, height), CV_64FC1);
		//cv::Mat J(cv::Size(width, height), CV_8U);
		
		while (abs(sigma_n - preSigma_n)>0.1){
			//while (1){
			preSigma_n = sigma_n;
			sigma_temp1 = 0;
			sigma_temp2 = 0;

			for (int x = 0; x < width; x++){
				for (int y = 0; y < height; y++){
					cv::Point2i z(x, y);
					cv::Point2i pNeighbor(x + vx.pData[y*width + x], y + vy.pData[y*width + x]);
					if (pNeighbor.x < 0 || pNeighbor.y < 0 || pNeighbor.x >= width || pNeighbor.y >= height)continue;
					J = calcDiff(src_t.at<T>(z) ,src_f.at<T>(pNeighbor),i);
					alfa.at<double>(z) = exp(-J / (2 * sigma_n*sigma_n)) / (exp(-J / (2 * sigma_n*sigma_n)) + 0.5*sigma_n*pow(2 * PI, 0.5));
					double temp = alfa.at<double>(z);
					sigma_temp1 += J*J*alfa.at<double>(z);
					sigma_temp2 += alfa.at<double>(z);
				}
			}
			sigma_n = pow(sigma_temp1 / sigma_temp2, 0.5);
		}
		sigma_t[i] = sigma_n*sizeof(T);
	}
}

template<class T>
void NLM<T>::estimateNoise2(double *sigma_t, cv::Mat src_t){
	//double sigma_t[3];
	for (int i = 0; i < sizeof(T); i++){
		double sigma_n = 1;
		double J;
		double sigma_temp1, sigma_temp2, preSigma_n = 0;
		//cv::Mat J(cv::Size(width, height), CV_8U);

		//while (abs(sigma_n - preSigma_n)>0.1){
			//while (1){
			preSigma_n = sigma_n;
			sigma_temp1 = 0;
			sigma_temp2 = 0;
			int count = 0;
			for (int x = 0; x < width; x++){
				for (int y = 0; y < height; y++){
					cv::Point2i z(x, y);
					//cv::Point2i pNeighbor(x + vx.pData[y*width + x], y + vy.pData[y*width + x]);
					//if (pNeighbor.x < 0 || pNeighbor.y < 0 || pNeighbor.x >= width || pNeighbor.y >= height)continue;
					J = getPixelValue(src_t.at<T>(z),i); //calcDiff(src_t.at<T>(z), src_f.at<T>(pNeighbor), i);
					sigma_temp1 += J;
					sigma_temp2 += J*J;
					count++;
				}
			}
			sigma_temp1 /= count;
			sigma_temp2 /= count;
			sigma_n =  pow(sigma_temp2-sigma_temp1*sigma_temp1,0.5);
		//}
		sigma_t[i] = sigma_n;
	}
}

