#include"NLM.h"
#include <windows.h> 

NLM::NLM(int H,int K,int lenPatch,vector<Mat>frames){
	this->H = H;
	this->K = K;
	this->S = (lenPatch - 1) / 2;
	this->frames = frames;
	this->sigma_p = S/2.0;
	this->sigma_t = 5;
	this->gama = 0.9;
}

double NLM::NLM_Estimate(Point3i p){
	double I=0;
	double Z = 0;
	for (int i = p.z - H; i <= p.z + H; i++){
		AKNN mAKNN(frames[i], frames[p.z]);
		vector<NeighborPatch> NNF;
		for (int ii = 0; ii < K; ii++){
			NNF.push_back(NeighborPatch(Point2i(500,500),100));
		}
		//vector<NeighborPatch> NNF = mAKNN.getV(Point2i(p.x, p.y), K, 2*S+1);
		for (int j = 0; j < K; j++)
		{
			double Dw = weightedSSD(Point3i(NNF[j].p.x, NNF[j].p.y, i), p);
			double temp = pow(gama, abs(i - p.z)) * exp(-(Dw / (2 * sigma_t*sigma_t)));
			Z += temp;
			I += frames[i].at<double>(NNF[j].p) * temp;
		}
	}
	I /= Z;

	return I;
}




double NLM::weightedSSD(Point3i p,Point3i q){
	double D = 0;
	double D_norm = 0;
	for (int i = -S; i <= S; i++){
		for (int j = -S; j <= S; j++){
			D += pow(frames[p.z].at<double>(p.y + j, p.x + i) - frames[q.z].at<double>(p.y + j, p.x + i), 2)*exp(-(i*i + j*j ) / (2.0 * sigma_p*sigma_p));
			//D_norm += exp(-(i*i + j*j) / (2 * sigma_p*sigma_p));
		}
	}
	//D_norm = pow(S * 2 + 1, 2);
	D /= D_norm;
	return D;
}