# include "AKNN.h"

AKNN::AKNN(const Mat &src, ImgKNN &_knnf) 
: imgSrc(src),KNNF( _knnf)
{
	

	nSrcCols = src.cols;
	nSrcRows = src.rows;

	

	cout << "start new" << endl;

	
	NNF = new Point2i *[nSrcCols];
	offset = new double *[nSrcCols];
	KNNF = new KNN *[nSrcCols];
	for (int i = 0; i < nSrcCols; i++){
		NNF[i] = new Point2i[nSrcRows];
		KNNF[i] = new KNN[nSrcRows];
		offset[i] = new double[nSrcRows];
	}

	srand(time(0));
}
AKNN::~AKNN(){
	cout << "release" << endl;
	double startRelease = getTickCount();
//#pragma omp parallel for 
	for (int i = 0; i < nSrcCols; i++){
		delete[]NNF[i];
		delete[]KNNF[i];
		delete[]offset[i];
	}
	delete[]NNF;
	delete[]KNNF;
	delete[]offset;
	double endRelease = getTickCount();
	cout << "cleared " <<endRelease-startRelease<< endl;
}

int AKNN::getV(int K, int lenPatch){
	this->K = K;
	this->S = (lenPatch - 1) / 2;

	
	operation();
	return 0;
}

void AKNN::setDst(Mat &_dst){
	
	nDstCols = _dst.cols;
	nDstRows = _dst.rows;
	sigma = min(nDstCols, nDstRows) / 3.0;
	imgDst = _dst;
}


void AKNN::operation(){

	int maxIteration = 4;
	int odd;
	initation();
	for (int iter = 1; iter <= maxIteration; iter++){
		cout << "start iteration " << iter << endl;
		odd = iter % 2;
		
		//if (odd){
		//	for (int x = S + 1; x <= nSrcCols - S - 3; x++){
		//		for (int y = S + 1; y <= nSrcRows - S - 3; y++){
		//			//cout << "progagation..." << endl;
		//			progagation(Point(x, y), odd);
		//			//cout << "randomSearch..." << endl;
		//			randomSearch(Point2i(x, y), NNF[x][y]);
		//		}
		//	}
		//}
		//else{
		//	for (int x = nSrcCols - S - 3; x >= S + 1; x--){
		//		for (int y = nSrcRows - S - 3; y >= S + 1; y--){
		//			progagation(Point(x, y), odd);
		//			randomSearch(Point2i(x, y), NNF[x][y]);
		//		}
		//	}
		//}
		if (odd){
			for (int x = 700; x <= 850; x++){
				for (int y = 50; y <= 200; y++){
					//cout << "progagation..." << endl;
					progagation(Point(x, y), odd);
					//cout << "randomSearch..." << endl;
					randomSearch(Point2i(x, y), NNF[x][y]);
				}
			}
		}
		else{
			for (int x = 850; x >= 700; x--){
				for (int y = 200; y >= 50; y--){
					progagation(Point(x, y), odd);
					randomSearch(Point2i(x, y), NNF[x][y]);
				}
			}
		}
	}
}



void AKNN::initation(){
	
//#pragma omp parallel for 
	for (int x = S; x < nSrcCols - S; x++){
		for (int y = S; y < nSrcRows - S; y++){
			Point2d n = generateNormal2dVector();
			Point2i p = (Point2i)(sigma*n) + Point2i(x, y);
			p.x = max(S+1, p.x);
			p.x = min(nDstCols - S - 2, p.x);
			p.y = max(S+1, p.y);
			p.y = min(nDstRows - S - 2, p.y);
			NNF[x][y] = p;
			calculateDistance(p, Point2i(x, y));
			offset[x][y] = calculateDistance(p, Point2i(x, y));
			KNNF[x][y].push_back(NeighborPatch(p, offset[x][y]));
		}
	}
	
}

void AKNN::progagation(Point2i patch, int odd){
	int x = patch.x;
	int y = patch.y;
	
	Rect nonLappingDstRect;
	Rect nonLappingSrcRect;
	Mat nonLapping(2*S+1,2*S+1,CV_64FC1,Scalar(0));
	
	if (odd){
		//up & left
		switch (getMinIndex(offset[x][y], offset[x][y - 1], offset[x - 1][y])){
			
		case 2:
			///////////////
			//    up
			///////////////
			if (NNF[x][y - 1].y > nDstRows - S - 2)break;

			NNF[x][y] = NNF[x][y-1];
			NNF[x][y].y += 1;

			nonLappingSrcRect=Rect(x-S ,y - S - 1 ,2 * S + 1, 1);
			
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.row(0));
			nonLappingSrcRect.y = y + S;
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.row(1));

			nonLappingDstRect = Rect(NNF[x][y].x - S, NNF[x][y].y - S - 1, 2 * S + 1, 1);
			imgDst(nonLappingDstRect).copyTo(nonLapping.row(2));
			nonLappingDstRect.y = NNF[x][y].y + S;
			imgDst(nonLappingDstRect).copyTo(nonLapping.row(3));

			offset[x][y] = offset[x][y-1] 
				- calculateDistance(nonLapping.row(2), nonLapping.row(0))
				+ calculateDistance(nonLapping.row(3), nonLapping.row(1));

			if ((offset[x][y] - calculateDistance(NNF[x][y], Point2i(x, y))) != 0){
				cout << "up" << endl;
			}

			KNNF[patch.x][patch.y].insert(KNNF[patch.x][patch.y].begin(), NeighborPatch(NNF[x][y], offset[x][y]));
			KNNF[patch.x][patch.y].pop_back();
			break;
			
		case 3:
			///////////////
			//    left
			///////////////
			if (NNF[x-1][y].x > nDstCols - S - 2)break;

			NNF[x][y] = NNF[x - 1][y];
			NNF[x][y].x += 1;


			nonLappingSrcRect = Rect(x - S - 1,y - S,1,2 * S + 1);
			
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.col(0));
			nonLappingSrcRect.x = x + S;
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.col(1));

			nonLappingDstRect = Rect(NNF[x][y].x - S - 1,NNF[x][y].y - S,1,2 * S + 1);

			imgDst(nonLappingDstRect).copyTo(nonLapping.col(2));
			nonLappingDstRect.x = NNF[x][y].x + S;
			imgDst(nonLappingDstRect).copyTo(nonLapping.col(3));

			offset[x][y] = offset[x - 1][y]
				- calculateDistance(nonLapping.col(2), nonLapping.col(0))
				+ calculateDistance(nonLapping.col(3), nonLapping.col(1));
			if ((offset[x][y] - calculateDistance(NNF[x][y], Point2i(x, y))) != 0){
				cout << "left" << endl;
			}
			
			KNNF[patch.x][patch.y].insert(KNNF[patch.x][patch.y].begin(), NeighborPatch(NNF[x][y], offset[x][y]));
			KNNF[patch.x][patch.y].pop_back();


			break;
			
		default:
			break;
		}
	}
	else{
		//down & right
		switch (getMinIndex(offset[x][y], offset[x][y + 1], offset[x + 1][y])){
		case 2:
			///////////////
			//    down
			///////////////
			if (NNF[x][y+1].y < S+1)break;
			NNF[x][y] = NNF[x][y + 1];
			NNF[x][y].y -= 1;




			nonLappingSrcRect = Rect(x - S, y + S + 1, 2 * S + 1, 1);
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.row(0));
			nonLappingSrcRect.y = y - S;
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.row(1));

			nonLappingDstRect = Rect(NNF[x][y].x - S, NNF[x][y].y + S + 1, 2 * S + 1, 1);
			imgDst(nonLappingDstRect).copyTo(nonLapping.row(2));
			nonLappingDstRect.y = NNF[x][y].y - S;
			imgDst(nonLappingDstRect).copyTo(nonLapping.row(3));

			offset[x][y] = offset[x][y + 1]
				- calculateDistance(nonLapping.row(2), nonLapping.row(0))
				+ calculateDistance(nonLapping.row(3), nonLapping.row(1));

			KNNF[patch.x][patch.y].insert(KNNF[patch.x][patch.y].begin(), NeighborPatch(NNF[x][y], offset[x][y]));
			KNNF[patch.x][patch.y].pop_back();

			if ((offset[x][y] - calculateDistance(NNF[x][y], Point2i(x, y))) != 0){
				cout << "down" << endl;
			}
			break;
		case 3:
			///////////////
			//    right
			///////////////


			if (NNF[x+1][y ].x < S + 1)break;
			NNF[x][y] = NNF[x + 1][y];
			NNF[x][y].x -= 1;


			nonLappingSrcRect = Rect(x + S + 1, y - S, 1, 2 * S + 1);
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.col(0));
			nonLappingSrcRect.x = x - S;
			imgSrc(nonLappingSrcRect).copyTo(nonLapping.col(1));

			nonLappingDstRect = Rect(NNF[x][y].x + S + 1, NNF[x][y].y - S, 1, 2 * S + 1);
			imgDst(nonLappingDstRect).copyTo(nonLapping.col(2));
			nonLappingDstRect.x = NNF[x][y].x - S;
			imgDst(nonLappingDstRect).copyTo(nonLapping.col(3));



			offset[x][y] = offset[x + 1][y]
				- calculateDistance(nonLapping.col(2), nonLapping.col(0))
				+ calculateDistance(nonLapping.col(3), nonLapping.col(1));

			KNNF[patch.x][patch.y].insert(KNNF[patch.x][patch.y].begin(), NeighborPatch(NNF[x][y], offset[x][y]));
			KNNF[patch.x][patch.y].pop_back();

			if ((offset[x][y] - calculateDistance(NNF[x][y], Point2i(x, y))) != 0){
				cout << nonLapping << endl;
				cout << offset[x][y] << '\t' << calculateDistance(NNF[x][y], Point2i(x, y)) << endl;
				cout << NNF[x][y] << '\t' << NNF[x+1][y] << endl;
				cout << "right" << endl;
			}
			break;
		default:

			break;
		}
	}
	
	
}

void AKNN::randomSearch(Point2i pSrc,Point2i nnf){
	int M = min(log2(sigma), K*1.0);
	Point2d n;
	Point2i v, p;
	double distance;
	//KNN nP;


	for (int i = 1; i <= M; i++){
		n = generateNormal2dVector();
		v = sigma*pow(ALPHA, i)*n;
		p = nnf + v;

		p.x = max(S, p.x);
		p.x = min(nDstCols - S - 1, p.x);
		p.y = max(S, p.y);
		p.y = min(nDstRows - S - 1, p.y);

		distance = calculateDistance(p, pSrc);
		handleQueue(KNNF[pSrc.x][pSrc.y], NeighborPatch(p, distance));
		if (distance < offset[pSrc.x][pSrc.y])nnf = p;
		
	}
	offset[pSrc.x][pSrc.y] = KNNF[pSrc.x][pSrc.y][0].distance;
	NNF[pSrc.x][pSrc.y] = KNNF[pSrc.x][pSrc.y][0].p;


	
	return;
}

double AKNN::calculateDistance(Point2i pDst, Point2i pSrc){
	Mat matDst = imgDst(Rect(pDst.x - S, pDst.y - S, 2 * S + 1, 2 * S + 1));
	Mat matSrc = imgSrc(Rect(pSrc.x - S, pSrc.y - S, 2 * S + 1, 2 * S + 1));
	return calculateDistance(matDst, matSrc);
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

void AKNN::handleQueue(KNN &knn, NeighborPatch mNeighborPatch){
	if (knn.size() == 0){
		knn.push_back(mNeighborPatch);
		return;
	}
	int low = 0, high = knn.size() - 1;
	int middle;
	while (low<=high){
		middle = (low + high) / 2;
		if (mNeighborPatch.distance == knn[middle].distance){
			break;
		}
		if (mNeighborPatch.distance > knn[middle].distance){
			low = middle + 1;
			middle += 1;
		}
		else 
			high = middle - 1;
	}
	knn.insert(knn.begin() + middle, mNeighborPatch);
	while (knn.size() > K)knn.pop_back();                 //保证K个最邻
	return;
}

Point2d AKNN::generateNormal2dVector(){
	double x = rand() % NUMMOD - NUMMOD / 2;
	double y = rand() % NUMMOD - NUMMOD / 2;
	double value = pow(x*x + y*y,0.5);
	x /= value;
	y /= value;
	return Point2d(x, y);
}


int AKNN::getMinIndex(double a, double b, double c){
	if (a<=b){
		if (a <= c)return 1;
		return 3;
	}
	else{
		if (b <= c)return 2;
		return 3;
	}
}