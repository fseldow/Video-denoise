#include"VideoDenoisingME.h"

VideoDenoisingME::VideoDenoisingME(){

}

int VideoDenoisingME::processing(vector<cv::Mat>srcFrames, vector<cv::Mat>&dstFrames, int K, int temporalWindowSize, int patchWindowSize)
{
	int H = temporalWindowSize / 2;
	temporalWindowSize = H * 2 + 1;
	int S = patchWindowSize / 2;
	patchWindowSize = S * 2 + 1;
	int width = srcFrames[0].cols;
	int height = srcFrames[0].rows;

	int border = S + 2;

	int videoSize = srcFrames.size();
	vector<cv::Mat>operateFrames;          //vector mat keeps newest 2H+1 frames
	vector<DImage>vxFlow;                  //vector DImage keeps newest 2H optical flow data for x dir
	vector<DImage>vyFlow;                  //vector DImage keeps newest 2H optical flow data for y dir

	//parameter for optical flow calculation
	double alpha = 0.012;
	double ratio=0.7;
	int minWidth = 20;
	int nOuterFPIterations = 7;
	int nInnerFPIterations = 1;
	int nCGIterations = 30;

	

	double start = GetTickCount();

	//calculate the first 2H optical flow data
	for (int n_frame = 0; n_frame < temporalWindowSize; n_frame++){
		cout << "frame : " << n_frame + 1 << endl;
		
		cv::Mat mExtentMat;
		extentMat(srcFrames[n_frame], mExtentMat, border);
		operateFrames.push_back(mExtentMat);

		
		if (n_frame>0){
			DImage pre, cur, warp, vx, vy;
			mat2DImage(operateFrames[n_frame - 1], pre);
			mat2DImage(operateFrames[n_frame], cur);
			OpticalFlow::Coarse2FineFlow(vx, vy, warp, pre, cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nCGIterations);
			double a=vx.pData[308100];
			vxFlow.push_back(vx);
			vyFlow.push_back(vy);
		}
	}
	//handle the first 2H+1 frames
	{
		cv::Mat resultTemp;                    //denoised pic
		videoDenoising(operateFrames, resultTemp, vxFlow, vyFlow, K, temporalWindowSize, patchWindowSize);
		cv::Mat result;
		centerMat(resultTemp, result, border);
		dstFrames.push_back(result);
		/*cv::imwrite("resultTest.jpg", result);
		cv::imshow("test", result);
		cv::waitKey();*/

	}

	double end = GetTickCount();
	cout << end - start << endl;
	
	for (int n_frame = 0; n_frame < videoSize - temporalWindowSize; n_frame++){
		cout <<"frame:"<< n_frame + temporalWindowSize+1 << endl;
 		operateFrames.erase(operateFrames.begin());
		cv::Mat mExtentMat;
		extentMat(srcFrames[n_frame + temporalWindowSize], mExtentMat, border);
		operateFrames.push_back(mExtentMat);

		DImage pre, cur, warp,vx,vy;
		mat2DImage(operateFrames[operateFrames.size() - 2], pre);
		mat2DImage(operateFrames[operateFrames.size() - 1], cur);
		OpticalFlow::Coarse2FineFlow(vx, vy, warp, pre, cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nCGIterations);
		vxFlow.erase(vxFlow.begin());
		vxFlow.push_back(vx);
		vyFlow.erase(vyFlow.begin());
		vyFlow.push_back(vy);

		cv::Mat resultTemp;                    //denoised pic
		videoDenoising(operateFrames, resultTemp, vxFlow, vyFlow, K, temporalWindowSize, patchWindowSize);
		cv::Mat result;
		centerMat(resultTemp, result, border);
		resultTemp.release();
		dstFrames.push_back(result);
	}
	return 0;
}

void VideoDenoisingME::videoDenoising(vector<cv::Mat>framesSrc, cv::Mat&framesOut, vector<DImage>vxFlow, vector<DImage>vyFlow, int _K, int temporalWindowSize, int patchWindowSize){
	if (vxFlow.size() != temporalWindowSize - 1 || vyFlow.size() != temporalWindowSize - 1)
		CV_Error(CV_StsBadArg,
		"optical flow vector must be size of temporalWindowSize-1");
	if (framesSrc.size() != temporalWindowSize)
		CV_Error(CV_StsBadArg,
		"src frames vector must be size of temporalWindowSize");
	if (temporalWindowSize%2==0)
		CV_Error(CV_StsBadArg,
		"temporalWindowSize must be odd");
	switch (framesSrc[0].type())
	{
	case CV_8U:
	{
				  NLM<uchar> mNLM(framesSrc, framesOut, _K, temporalWindowSize, patchWindowSize, vxFlow, vyFlow);
				  mNLM.operation();
				  break;
	}
	case CV_8UC2:
	{
					NLM<cv::Vec2b> mNLM(framesSrc, framesOut, _K, temporalWindowSize, patchWindowSize, vxFlow, vyFlow);
					mNLM.operation();
					break;
	}
	case CV_8UC3:
	{
					NLM<cv::Vec3b> mNLM(framesSrc, framesOut, _K, temporalWindowSize, patchWindowSize, vxFlow, vyFlow);
					mNLM.operation();
					break;
	}
	default:
		CV_Error(CV_StsBadArg,
			"Unsupported image format! Only CV_8UC1, CV_8UC2 and CV_8UC3 are supported");
		break;
	}
}


void VideoDenoisingME::videoDenoising(vector<cv::Mat>framesSrc, cv::Mat&framesOut, int _K, int temporalWindowSize, int patchWindowSize){
	
	int H = temporalWindowSize / 2;
	temporalWindowSize = H * 2 + 1;
	int S = patchWindowSize / 2;
	patchWindowSize = S * 2 + 1;
	int width = framesSrc[0].cols;
	int height = framesSrc[0].rows;
	int border = S + 2;

	vector<DImage>vxFlow;                  //vector DImage keeps newest 2H optical flow data for x dir
	vector<DImage>vyFlow;                  //vector DImage keeps newest 2H optical flow data for y dir

	//parameter for optical flow calculation
	double alpha = 0.012;
	double ratio = 0.7;
	int minWidth = 20;
	int nOuterFPIterations = 7;
	int nInnerFPIterations = 1;
	int nCGIterations = 30;

	if (framesSrc.size() != temporalWindowSize)
		CV_Error(CV_StsBadArg,
		"src frames vector must be size of temporalWindowSize");

	vector<cv::Mat> operateMat;
	for (int n_frame = 0; n_frame < temporalWindowSize; n_frame++){
		cv::Mat temp;
		extentMat(framesSrc[n_frame], temp, border);
		operateMat.push_back(temp);
	}
	

	//calculate  2H optical flow data
	for (int n_frame = 0; n_frame < temporalWindowSize; n_frame++){
		if (n_frame>0){
			DImage pre, cur, warp, vx, vy;
			mat2DImage(operateMat[n_frame - 1], pre);
			mat2DImage(operateMat[n_frame], cur);
			OpticalFlow::Coarse2FineFlow(vx, vy, warp, pre, cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nCGIterations);
			vxFlow.push_back(vx);
			vyFlow.push_back(vy);
		}
	}
	//handle the  2H+1 frames
	cv::Mat resultTemp;
	videoDenoising(operateMat, resultTemp, vxFlow, vyFlow, _K, temporalWindowSize, patchWindowSize);
	centerMat(resultTemp, framesOut, border);
		

	
}

void VideoDenoisingME::extentMat(cv::Mat src, cv::Mat &dst, int extentEdge){
	/*if(dst.empty())dst.create(src.rows+2*extentEdge,src.cols+2*extentEdge,src.type());
	else dst.resize(src.rows + 2 * extentEdge, src.cols + 2 * extentEdge);*/
	cv::copyMakeBorder(src, dst, extentEdge, extentEdge, extentEdge, extentEdge, cv::BORDER_REFLECT_101, cv::Scalar::all(0));
}

void VideoDenoisingME::centerMat(cv::Mat src, cv::Mat &dst, int extentEdge){
	if (dst.empty())dst.create(src.rows - 2* extentEdge, src.cols -2* extentEdge, src.type());
	else dst.resize(src.rows - 2 * extentEdge, src.cols - 2 * extentEdge);

	cv::Rect center(extentEdge, extentEdge, dst.cols, dst.rows);
	src(center).copyTo(dst);
}


