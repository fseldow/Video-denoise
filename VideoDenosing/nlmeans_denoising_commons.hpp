#ifndef __NLMEANS_DENOISING_COMMONS_HPP__
#define __NLMEANS_DENOISING_COMMONS_HPP__

#include "cv.h"
#include "highgui.h"
using namespace std;
//using namespace cv;

template <typename T> static inline int calcDistance(const T a, const T b);

template <> inline int calcDistance(const uchar a, const uchar b) {
	return (a - b) * (a - b);
}

template <> inline int calcDistance(const cv::Vec2b a, const cv::Vec2b b) {
	return (a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]);
}

template <> inline int calcDistance(const cv::Vec3b a, const cv::Vec3b b) {
	return (a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]) + (a[2] - b[2])*(a[2] - b[2]);
}


template <typename T> static inline int calcDiff(const T a, const T b,int index);

template <> inline int calcDiff(const uchar a, const uchar b, int index) {
	return (a-b);
}

template <> inline int calcDiff(const cv::Vec2b a, const cv::Vec2b b, int index) {
	return (a[index]-b[index]);
}

template <> inline int calcDiff(const cv::Vec3b a, const cv::Vec3b b, int index) {
	return (a[index] - b[index]);
}



template <typename T> static inline int getPixelValue(const T a,  int index);

template <> inline int getPixelValue(const uchar a,  int index) {
	return a;
}

template <> inline int getPixelValue(const cv::Vec2b a,  int index) {
	return (a[index] );
}

template <> inline int getPixelValue(const cv::Vec3b a,  int index) {
	return (a[index] );
}



template <typename T> static inline void incWithWeight(double* estimation, double* weight, T p);
template <> inline void incWithWeight(double* estimation, double* weight, uchar p) {
	estimation[0] += weight[0] * p;
}

template <> inline void incWithWeight(double* estimation, double*weight, cv::Vec2b p) {
	estimation[0] += weight[0] * p[0];
	estimation[1] += weight[1] * p[1];
}

template <> inline void incWithWeight(double* estimation, double* weight, cv::Vec3b p) {
	estimation[0] += weight[0] * p[0];
	estimation[1] += weight[1] * p[1];
	estimation[2] += weight[2] * p[2];
}






template <typename T> static inline T saturateCastFromArray(double* estimation);

template <> inline uchar saturateCastFromArray(double* estimation) {
	return cv::saturate_cast<uchar>(estimation[0]);
}

template <> inline cv::Vec2b saturateCastFromArray(double* estimation) {
	cv::Vec2b res;
	res[0] = cv::saturate_cast<uchar>(estimation[0]);
	res[1] = cv::saturate_cast<uchar>(estimation[1]);
	return res;
}

template <> inline cv::Vec3b saturateCastFromArray(double* estimation) {
	cv::Vec3b res;
	res[0] = cv::saturate_cast<uchar>(estimation[0]);
	res[1] = cv::saturate_cast<uchar>(estimation[1]);
	res[2] = cv::saturate_cast<uchar>(estimation[2]);
	return res;
}


template <typename T> static inline T removeErrBlack(T ori,T est);

template <> inline uchar removeErrBlack(uchar ori, uchar est) {
	if (est == 0 && ori != 0)return ori;
	return est;
}

template <> inline cv::Vec2b removeErrBlack(cv::Vec2b ori, cv::Vec2b est) {
	cv::Vec2b res;
	for (int i = 0; i < 2; i++){
		if (est[i] == 0 && ori[i] != 0)res[i]=ori[i];
		else res[i] = est[i];
	}
	return res;
}

template <> inline cv::Vec3b removeErrBlack(cv::Vec3b ori, cv::Vec3b est) {
	cv::Vec3b res;
	for (int i = 0; i < 3; i++){
		if (est[i] == 0 && ori[i] != 0)res[i] = ori[i];
		else res[i] = est[i];
	}
	return res;
}


#endif