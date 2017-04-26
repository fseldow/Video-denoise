#include"NLM.h"
#include <windows.h> 

NLM::NLM(int _H, int _K, int lenPatch, vector<cv::Mat>&_frames, cv::Mat&_dst) :frames(_frames), dst(_dst){
	this->H = _H;
	this->K = _K;
	this->S = (lenPatch - 1) / 2;
	this->sigma_p = S/2.0;
	this->gama = 0.9;
	int w = _frames[0].cols;
	int h = _frames[0].rows;
	board = cv::Rect(S + 1, S + 1, w - 2 * S - 2, h - 2 * S - 2);
	board = cv::Rect(700, 50, 100, 100);

	double start, end;
	start = GetTickCount();
	if (dst.empty()){
				//¿½±´
//#pragma omp parallel for
		frames[H].copyTo(dst);
	}
	end = GetTickCount();
	cout << "ff" << end - start<<endl;
}

void NLM::operator()(const cv::Range& range) const{
	/*for (int i = range.start; i < range.end; i++){
		for (int j = 50; j < 100; j++){
			int a = pow(2, 3);
		}
	}*/
	for (int f = H; f < frames.size() - H; f++){
		double sigma_t = 0;//getSigma_t(frames[f], frames[f + 1]);
		for (int y = range.start; y < range.end; y++){
			for (int x = 300; x < 550; x++){
			//dst[f].at<double>(y, x) = NLM_Estimate(Point3i(x, y, f),sigma_t);
			}
		}
	}
}

//void NLM::setSigma_t(double m_sigma_t)const{
//	sigma_t = sigma_t;
//}

double NLM::NLM_Estimate(cv::Point3i p, double sigma_t, vector<ImgKNN> vKNN)const{
	double I=0;
	double Z = 0;
	for (int i = p.z - H; i <= p.z + H; i++){

		vector<NeighborPatch> NNF=(vKNN[i])[p.x][p.y];
		for (int j = 0; j < K; j++)
		{
			double Dw = weightedSSD(cv::Point3i(NNF[j].p.x, NNF[j].p.y, i), p);
			double temp = pow(gama, abs(i - p.z)) * exp(-(Dw / (2 * sigma_t*sigma_t)));
			Z += temp;
			I += frames[i].at<double>(NNF[j].p) * temp;
		}
	}
	I /= Z;
	double ori = frames[p.z].at<double>(p.y, p.x);
	return I;
}




double NLM::weightedSSD(cv::Point3i p, cv::Point3i q)const{
	double D = 0;
	double D_norm = 0;
	for (int i = -S; i <= S; i++){
		for (int j = -S; j <= S; j++){
			double temp = exp(-(i*i + j*j) / (2.0 * sigma_p*sigma_p));
			D += pow(frames[p.z].at<double>(p.y + j, p.x + i) - frames[q.z].at<double>(p.y + j, p.x + i), 2)*temp;
			D_norm += temp;
		}
	}
	D /= D_norm;
	return D;
}

double NLM::getSigma_t(cv::Mat src_t, cv::Mat src_f,KNN z,DImage vx,DImage vy)const{
	int width = 1260;
	double sigma_n = 20;
	int rows = src_t.rows;
	int cols = src_t.cols;
	double J;
	double sigma_temp1,sigma_temp2,preSigma_n=0;
	cv::Mat alfa(cv::Size(cols, rows), CV_64FC1);

	while (abs(sigma_n-preSigma_n)>0.1){
	//while (1){
		preSigma_n = sigma_n;
		sigma_temp1 = 0;
		sigma_temp2 = 0;
		for (int i = 0; i < z.size(); i++){
			cv::Point2i pNeighbor(z[i].p.x + vx.pData[z[i].p.y*width + z[i].p.x], z[i].p.y + vy.pData[z[i].p.y*width + z[i].p.x]);
			J = src_t.at<double>(z[i].p) - src_f.at<double>(pNeighbor);
			alfa.at<double>(z[i].p) = exp(-J / (2 * sigma_n*sigma_n)) / (exp(-J / (2 * sigma_n*sigma_n)) + 0.5*sigma_n*pow(2 * PI, 0.5));
			//int test = alfa.at<double>(i, j);
			sigma_temp1 += J*J*alfa.at<double>(z[i].p);
			sigma_temp2 += alfa.at<double>(z[i].p);
		}
		/*for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				J = src_t.at<double>(i, j) - src_f.at<double>(i, j);
				alfa.at<double>(i, j) = exp(-J / (2 * sigma_n*sigma_n)) / (exp(-J / (2 * sigma_n*sigma_n))+0.5*sigma_n*pow(2*PI,0.5));
				int test = alfa.at<double>(i, j);
				sigma_temp1 += J*J*alfa.at<double>(i, j);
				sigma_temp2 += alfa.at<double>(i, j);
			}
		}*/
		sigma_n = pow(sigma_temp1 / sigma_temp2, 0.5);
	}
	return sigma_n;
}