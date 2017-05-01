#include"VideoDenoisingME.h"

VideoDenoisingME::VideoDenoisingME(){

}

int VideoDenoisingME::processing(vector<cv::Mat>srcFrames, vector<cv::Mat>&dstFrames, int K, int temporalWindowSize, int patchWindowSize)
{
	cv::VideoCapture capture("E:\\C++\\video1_poor.mp4");
	double fps = capture.get(CV_CAP_PROP_FPS);
	//获得原始视频的高度和宽度
	cv::Size size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	///创建一个视频文件参数分别表示  新建视频的名称 视频压缩的编码格式 新建视频的帧率 新建视频的图像大小
	cv::VideoWriter writer("E:\\C++\\poor_result.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, size);

	this->K = K;
	this->H = temporalWindowSize / 2;
	temporalWindowSize = H * 2 + 1;
	this->S = patchWindowSize / 2;
	patchWindowSize = S * 2 + 1;
	this->width = srcFrames[0].cols;
	this->height = srcFrames[0].rows;

	int videoSize = srcFrames.size();
	vector<cv::Mat>operateFrames;
	vector<DImage>vxFlow;
	vector<DImage>vyFlow;


	double alpha=1, ratio=0.7;
	int minWidth=30, nOuterFPIterations=15, nInnerFPIterations=1, nCGIterations=40;

	cv::Mat resultTemp;

	double start = GetTickCount();
	for (int n_frame = 0; n_frame < temporalWindowSize; n_frame++){
		cout << "frame : " << n_frame + 1 << endl;
		operateFrames.push_back(srcFrames[n_frame]);
		
		if (n_frame>0){
			DImage pre, cur, warp, vx, vy;
			mat2DImage(srcFrames[n_frame - 1], pre);
			mat2DImage(srcFrames[n_frame], cur);
			OpticalFlow::Coarse2FineFlow(vx, vy, warp, pre, cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nCGIterations);
			vxFlow.push_back(vx);
			vyFlow.push_back(vy);
		}
	}
	{
		videoDenoising(operateFrames, resultTemp, vxFlow, vyFlow, K, temporalWindowSize, patchWindowSize);
		dstFrames.push_back(resultTemp);
		writer.write(resultTemp);
	}
	double end = GetTickCount();
	cout << end - start << endl;
	cv::imshow("test", resultTemp);
	cv::waitKey();
	for (int n_frame = 0; n_frame < videoSize - temporalWindowSize; n_frame++){
 		operateFrames.erase(operateFrames.begin());
		operateFrames.push_back(srcFrames[n_frame + temporalWindowSize]);

		DImage pre, cur, warp,vx,vy;
		mat2DImage(operateFrames[operateFrames.size() - 2], pre);
		mat2DImage(operateFrames[operateFrames.size() - 1], cur);
		OpticalFlow::Coarse2FineFlow(vx, vy, warp, pre, cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nCGIterations);
		vxFlow.erase(vxFlow.begin());
		vxFlow.push_back(vx);
		vyFlow.erase(vyFlow.begin());
		vyFlow.push_back(vy);

		videoDenoising(operateFrames, resultTemp, vxFlow, vyFlow, K, temporalWindowSize, patchWindowSize);
		dstFrames.push_back(resultTemp);
		writer.write(resultTemp);
	}
	return 0;
}

void VideoDenoisingME::videoDenoising(vector<cv::Mat>framesSrc, cv::Mat&framesOut, vector<DImage>vxFlow, vector<DImage>vyFlow, int _K, int temporalWindowSize, int patchWindowSize){
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



