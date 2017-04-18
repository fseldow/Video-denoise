# include "AKNN.h"

AKNN::AKNN(const Mat img,const Mat imgSrc){
	this->img = img;
	this->imgSrc = imgSrc;
	
	nImgCols = img.cols;
	nImgRows = img.rows;
}

vector<NeighborPatch> AKNN::getV(Point2i pPatch, int K, int lenPatch){
	this->K = K;
	this->S = (lenPatch - 1) / 2;
	Rect patchSrcRect(pPatch.x - S, pPatch.y - S, lenPatch, lenPatch);
	this->patch = imgSrc(patchSrcRect);
	this->pPatch = pPatch;
	patchSurroungding[0] = imgSrc(Rect(pPatch.x, pPatch.y - S - 1, lenPatch, 1));
	patchSurroungding[1] = imgSrc(Rect(pPatch.x + S + 1, pPatch.y, 1, lenPatch));
	patchSurroungding[2] = imgSrc(Rect(pPatch.x, pPatch.y + S + 1, lenPatch, 1));
	patchSurroungding[3] = imgSrc(Rect(pPatch.x - S - 1, pPatch.y , 1,lenPatch));
	operation();
	return neighbors;
}

void AKNN::initation(){
	//NeighborPatch mNeighborPatch(Point2i(-1,-1),-1);
	//for (int i = 0; i < K;i++)neighbors.push_back(mNeighborPatch);
	sigma = min(nImgRows, nImgCols) / 3.0;
}

void AKNN::progagation(Point2i pNeighborPatch, int iteration){

	//��ֹ���
	pNeighborPatch.x = max(S + 2, pNeighborPatch.x);
	pNeighborPatch.x = min(nImgCols-S-2, pNeighborPatch.x);

	pNeighborPatch.y = max(S + 2, pNeighborPatch.y);
	pNeighborPatch.y = min(nImgRows - S - 2, pNeighborPatch.y);

	//������������
	int odd = iteration % 2;

	double mDistance,tempDistance;
	mDistance = calculateDistance(pNeighborPatch, patch);
	tempDistance = mDistance;
	handleQueue(NeighborPatch(pNeighborPatch, mDistance));

	Mat nonLapping[2];      //nearest neighbor areas with 1 pixel offset

	if (odd)                //scanline order
	{
		//up 
		nonLapping[0] = img(Rect(pNeighborPatch.x - S, pNeighborPatch.y - S - 1, 2 * S + 1, 1));
		nonLapping[1] = img(Rect(pNeighborPatch.x - S, pNeighborPatch.y + S, 2 * S + 1, 1));
		mDistance = tempDistance + calculateDistance(nonLapping[0], patchSurroungding[0]) - calculateDistance(nonLapping[1], patch(Rect(0, 2 * S, 2 * S + 1, 1)));
		handleQueue(NeighborPatch(pNeighborPatch + Point2i(0, -1), mDistance));
		//left
		nonLapping[0] = img(Rect(pNeighborPatch.x - S - 1, pNeighborPatch.y - S, 1, 2 * S + 1));
		nonLapping[1] = img(Rect(pNeighborPatch.x + S, pNeighborPatch.y - S, 1, 2 * S + 1));
		mDistance = tempDistance + calculateDistance(nonLapping[0], patchSurroungding[3]) - calculateDistance(nonLapping[1], patch(Rect(0, 0, 1, 2 * S + 1)));
		handleQueue(NeighborPatch(pNeighborPatch + Point2i(-1, 0), mDistance));
	}

	else                    //reverse scanline order
	{
		//right
		nonLapping[0] = img(Rect(pNeighborPatch.x + S + 1, pNeighborPatch.y - S, 1, 2 * S + 1));
		nonLapping[1] = img(Rect(pNeighborPatch.x - S, pNeighborPatch.y - S, 1, 2 * S + 1));
		mDistance = tempDistance + calculateDistance(nonLapping[0], patchSurroungding[1]) - calculateDistance(nonLapping[1], patch(Rect(2 * S, 0, 1, 2 * S + 1)));
		handleQueue(NeighborPatch(pNeighborPatch + Point2i(1, 0), mDistance));
		//down 
		nonLapping[0] = img(Rect(pNeighborPatch.x - S, pNeighborPatch.y + S + 1, 2 * S + 1, 1));
		nonLapping[1] = img(Rect(pNeighborPatch.x - S, pNeighborPatch.y - S, 2 * S + 1, 1));
		mDistance = tempDistance + calculateDistance(nonLapping[0], patchSurroungding[2]) - calculateDistance(nonLapping[1], patch(Rect(0, 0, 2 * S + 1, 1)));
		handleQueue(NeighborPatch(pNeighborPatch + Point2i(0, 1), mDistance));
	}
}

Point2i AKNN::randomSearch(int i){
	Point2d n = generateNormal2dVector();
	Point2i v = sigma*pow(ALPHA, i)*n;
	return v;
}

double AKNN::calculateDistance(Point2i p, Mat patch){
	Mat neighborPatch = img(Rect(p.x - S, p.y - S, 2 * S + 1, 2 * S + 1));
	return calculateDistance(patch, neighborPatch);
}

double AKNN::calculateDistance(Mat q, Mat p){
	double result = 0.0;
	for (int i = 0; i < q.rows; i++){
		for (int j = 0; j < q.cols; j++){
			result += pow(q.at<double>(i,j) - p.at<double>(i, j), 2);
		}
	}
	return result;
}

void AKNN::handleQueue(NeighborPatch mNeighborPatch){
	if (neighbors.size() ==0){ neighbors.push_back(mNeighborPatch); return; }
	int low = 0, high = neighbors.size()-1;
	int middle;
	while (low<=high){
		middle = (low + high) / 2;
		if (mNeighborPatch.distance == neighbors[middle].distance){
			break;
		}
		if (mNeighborPatch.distance > neighbors[middle].distance){
			low = middle + 1;
			middle += 1;
		}
		else 
			high = middle - 1;
	}
	neighbors.insert(neighbors.begin() + middle , mNeighborPatch);
	while (neighbors.size() > K)neighbors.pop_back();                 //��֤K������
	return;
}

Point2d AKNN::generateNormal2dVector(){
	srand(time(0));
	double x = rand() % NUMMOD - NUMMOD / 2;
	double y = rand() % NUMMOD - NUMMOD / 2;
	double value = pow(x*x + y*y,0.5);
	x /= value;
	y /= value;
	return Point2d(x, y);
}

void AKNN::operation(){
	//cout << "initating..." << endl;
	initation();
	int M = min(log2(sigma), K*1.0);
	//int M = 4;
	for (int i = 0; i <= M; i++){
		//cout << "iterate "<<i+1<<" ..." << endl;
		Point2i v_offset;
		//cout << "searching..." << endl;
		v_offset=randomSearch(i);
		//cout << "progagating..." << endl;
		progagation(v_offset + pPatch,i);
	}
}