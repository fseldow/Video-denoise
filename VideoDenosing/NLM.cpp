#include"NLM.h"
#include <windows.h> 

NLM::NLM(int _H, int _K, int lenPatch, vector<Mat>_frames, vector<Mat>&_dst) :frames(_frames),dst(_dst){
	this->H = _H;
	this->K = _K;
	this->S = (lenPatch - 1) / 2;
	this->sigma_p = S/2.0;
	this->gama = 0.9;

	double start, end;
	start = GetTickCount();
	if (dst.empty()){
				//¿½±´
		for (int f = 0; f < frames.size() ; f++){
			Mat temp;
			frames[f].copyTo(temp);
			dst.push_back(temp);
		}
	}
	end = GetTickCount();
	cout << "ff" << end - start<<endl;
}

void NLM::operator()(const Range& range) const{
	/*for (int i = range.start; i < range.end; i++){
		for (int j = 50; j < 100; j++){
			int a = pow(2, 3);
		}
	}*/
	for (int f = H; f < frames.size() - H; f++){
		double sigma_t= getSigma_t(frames[f], frames[f + 1]);
		for (int y = range.start; y < range.end; y++){
			
			for (int x = 300; x < 550; x++){
			//int N = 22500;
			//int y = 50+n/100;
			//int x = 300+n%100;
			//cout << " x " << x << " y " << y <<" n "<<n<< endl;
			dst[f].at<double>(y, x) = NLM_Estimate(Point3i(x, y, f),sigma_t);
			//cout << "finish: x " << x << " y " << y << endl;
			}
		}
	}
}

//void NLM::setSigma_t(double m_sigma_t)const{
//	sigma_t = sigma_t;
//}

double NLM::NLM_Estimate(Point3i p, double sigma_t)const{
	double I=0;
	double Z = 0;
	for (int i = p.z - H; i <= p.z + H; i++){
		AKNN mAKNN(frames[i], frames[p.z]);
		
		vector<NeighborPatch> NNF = mAKNN.getV(Point2i(p.x, p.y), K, 2*S+1);
		for (int j = 0; j < K; j++)
		{
			double Dw = weightedSSD(Point3i(NNF[j].p.x, NNF[j].p.y, i), p);
			double temp = pow(gama, abs(i - p.z)) * exp(-(Dw / (2 * sigma_t*sigma_t)));
			Z += temp;
			I += frames[i].at<double>(NNF[j].p) * temp;
		}
	}
	I /= Z;
	double ori = frames[p.z].at<double>(p.y, p.x);
	return I;
}




double NLM::weightedSSD(Point3i p,Point3i q)const{
	double D = 0;
	double D_norm = 0;
	for (int i = -S; i <= S; i++){
		for (int j = -S; j <= S; j++){
			double temp = exp(-(i*i + j*j) / (2.0 * sigma_p*sigma_p));
			D += pow(frames[p.z].at<double>(p.y + j, p.x + i) - frames[q.z].at<double>(p.y + j, p.x + i), 2)*temp;
			D_norm += temp;
		}
	}
	//D_norm = pow(S * 2 + 1, 2);
	D /= D_norm;
	//D /= (2 * S + 1)*(2 * S + 1);
	return D;
}

double NLM::getSigma_t(Mat src_t, Mat src_f)const{
	double sigma_n = 20;
	int rows = src_t.rows;
	int cols = src_t.cols;
	double J;
	double sigma_temp1,sigma_temp2,preSigma_n=0;
	Mat alfa(Size(cols,rows),CV_64FC1);

	while (abs(sigma_n-preSigma_n)>0.01){
	//while (1){
		preSigma_n = sigma_n;
		sigma_temp1 = 0;
		sigma_temp2 = 0;
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				J = src_t.at<double>(i, j) - src_f.at<double>(i, j);
				alfa.at<double>(i, j) = exp(-J / (2 * sigma_n*sigma_n)) / (exp(-J / (2 * sigma_n*sigma_n))+0.5*sigma_n*pow(2*PI,0.5));
				int test = alfa.at<double>(i, j);
				sigma_temp1 += J*J*alfa.at<double>(i, j);
				sigma_temp2 += alfa.at<double>(i, j);
			}
		}
		sigma_n = pow(sigma_temp1 / sigma_temp2, 0.5);
	}
	return sigma_n;
}