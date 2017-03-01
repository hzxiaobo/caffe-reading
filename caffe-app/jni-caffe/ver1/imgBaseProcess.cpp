#include "imgBaseProcess.h"

imgBaseProcess::imgBaseProcess(){
}

imgBaseProcess::~imgBaseProcess(){	
}

void imgBaseProcess::clear(){
}



Mat imgBaseProcess::regular(Mat mat, int length, int processType){
	try{
		if (mat.rows == 0 || mat.cols == 0 || length <= 0){
			std::cout << "in imgBaseProcess::regular, please check input paras  " << std::endl;
			Mat nullMat(0,0,CV_8UC1);
			return nullMat;
		}
		if (processType == 1){
			return regularSquare(mat, length);
		}else if (processType ==2){
			return regularSquareMidCut(mat,length);
		}else{
			return regularOrigin(mat, length);
		}
}catch (Exception* e){
	std::cout << "error in imgBaseProcess::regular : " << e << std::endl;
	Mat nullMat(0,0,CV_8UC1);
	return nullMat;
}
}

Mat imgBaseProcess::regularSquare(Mat mat, int length){
	try{
		mat = regularOrigin(mat,length);
		Mat opMat = Mat::ones(length, length,mat.type());
		opMat = opMat*255;
		if (mat.cols <= mat.rows){
			Rect r(0, 0, mat.cols-1, mat.rows-1);  // ָ��src �� ROI��ͼ������
			Mat dstroi = opMat(Rect(int((length - mat.cols)/2) ,0,r.width,r.height)); // �õ� dstָ��������ͼ�������
			mat(r).convertTo(dstroi, dstroi.type(), 1, 0); // ROI��ͼ��֮��ĸ���
			return opMat;
		}else{
			Rect r(0, 0, mat.cols-1, mat.rows-1);  // ָ��src �� ROI��ͼ������
			Mat dstroi = opMat(Rect(0,int((length - mat.rows)/2),r.width,r.height)); // �õ� dstָ��������ͼ�������
			mat(r).convertTo(dstroi, dstroi.type(), 1, 0); // ROI��ͼ��֮��ĸ���
			return opMat;
		}
	}catch(Exception e){
		Mat nullMat(0,0,CV_8UC1);
		return nullMat;
	}
}


Mat imgBaseProcess::regularOrigin(Mat mat, int length){
	try{
		if (mat.cols <= mat.rows){
			int newCols = int ( length * mat.cols / mat.rows  );
			if (newCols == 0){
				Mat nullMat(0,0,CV_8UC1);
				return nullMat;
			}
			resize(mat, mat, Size(newCols, length));
			return mat;
		}else{
			int newRows = int ( length * mat.rows / mat.cols  );
			if (newRows == 0){
				Mat nullMat(0,0,CV_8UC1);
				return nullMat;
			}
			resize(mat, mat, Size(length, newRows));
			return mat;
		}
	}catch(Exception e){
		Mat nullMat(0,0,CV_8UC1);
		return nullMat;
	}
}


Mat imgBaseProcess::regularSquareMidCut(Mat mat, int length){
	try{
		if (mat.cols >= mat.rows){
			int newCols = int ( length * mat.cols / mat.rows  );
			if (newCols == 0){
				Mat nullMat(0,0,CV_8UC1);
				return nullMat;
			}
			resize(mat, mat, Size(newCols, length));
			Rect r(int((mat.cols-mat.rows)/2), 0, length, length);  // ָ��src �� ROI��ͼ������
			Mat opMat = mat(r); // �õ� dstָ��������ͼ�������
			return opMat;
		}else{
			int newRows = int ( length * mat.rows / mat.cols  );
			if (newRows == 0){
				Mat nullMat(0,0,CV_8UC1);
				return nullMat;
			}
			resize(mat, mat, Size(length, newRows));
			Rect r(0,int((mat.rows-mat.cols)/2), length, length);  // ָ��src �� ROI��ͼ������
			Mat opMat = mat(r); // �õ� dstָ��������ͼ�������
			return opMat;
		}
	}catch(Exception e){
		Mat nullMat(0,0,CV_8UC1);
		return nullMat;
	}
}

vector<Mat> imgBaseProcess::regularSlice(Mat mat, int length, int size, int minStep){
	vector<Mat> opMats;
	try{
		if (mat.cols >= mat.rows){
			int newCols = int ( length * mat.cols / mat.rows  );
			if (newCols == 0){
				return opMats;
			}
			resize(mat, mat, Size(newCols, length));
			int dist = newCols - length;
			if (dist < 5){
				opMats.push_back(mat);
			}else if (dist<(size-1)*minStep){
				int stepSize = floor(dist/minStep);
				for (int i = 0; i < stepSize ; i++){
					Rect r((minStep*i),0,length, length);
					Mat subMat = mat(r);
					opMats.push_back(subMat);
				}
				Rect r((newCols - length),0,length, length);
				Mat subMat = mat(r);
				opMats.push_back(subMat);
			}else{
				int step = ceil(dist/(size-1));
				for(int i = 0 ; i < size-1 ; i++){
					Rect r((step*i),0,length, length);
					Mat subMat = mat(r);
					opMats.push_back(subMat);
				}
				Rect r((newCols - length),0,length, length);
				Mat subMat = mat(r);
				opMats.push_back(subMat);
			}
			return opMats;
		}else{
			int newRows = int ( length * mat.rows / mat.cols  );
			if (newRows == 0){
				return opMats;
			}
			resize(mat, mat, Size(length, newRows));
			int dist = newRows - length;
			if (dist < 5){
				opMats.push_back(mat);
			}else if (dist<(size-1)*minStep){
				int stepSize = floor(dist/minStep);
				for (int i = 0; i < stepSize ; i++){
					Rect r(0,(minStep*i),length, length);
					Mat subMat = mat(r);
					opMats.push_back(subMat);
				}
				Rect r(0,(newRows - length),length, length);
				Mat subMat = mat(r);
				opMats.push_back(subMat);
			}else{
				int step = ceil(dist/(size-1));
				for(int i = 0 ; i < size-1 ; i++){
					Rect r(0,(step*i),length, length);
					Mat subMat = mat(r);
					opMats.push_back(subMat);
				}
				Rect r(0,(newRows - length),length, length);
				Mat subMat = mat(r);
				opMats.push_back(subMat);
			}
			return opMats;
		}	
	}catch(Exception e){
		return opMats;
	}
}



//void testImgBaseProcess(){
//	cout <<"Test imgBaseProcess:" << endl;
//	string file = "D:/data/Image/dl/dl_test_folder_2k/porn/1392962103709_1d90fefca3844ec079eaaca1f146c670_56320908.JPEG";
//	//string file = "D:/data/Image/dl/dl_test_folder_2k/porn/1392962103486_d4ec4a3a840a7da62ba31d9f3601ee8b_614822850.JPEG";
//	Mat img = imread(file,1);
//	imshow("load image", img);
//	waitKey();
//
//	Mat originImg = imgBaseProcess::regular(img, 256, 0);
//	imshow("resize image", originImg);
//	waitKey();
//
//	Mat squareImg = imgBaseProcess::regular(img, 256, 1);
//	imshow("square image", squareImg);
//	waitKey();
//
//	Mat squareCutImg = imgBaseProcess::regular(img, 256, 2);
//	imshow("square image", squareCutImg);
//	waitKey();
//}


