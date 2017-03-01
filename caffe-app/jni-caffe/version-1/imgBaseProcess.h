#pragma once

#include "iostream"     
#include "cv.h"   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <vector>
//#include "../textDetect/textdetection.h"

using namespace cv;

//void testImgBaseProcess();

class imgBaseProcess{

public:
	//构造函数
	imgBaseProcess();

	// 析构函数
	~imgBaseProcess();

	void clear();
	
	/************************************************************************/
	/* 图像标准化，将图像转换为标准图像，有几种模式可选，0是原始模式，1是方块模式（按长边，多余部分补为白色），2是两端切除的方块模式                                                                    */
	/************************************************************************/
	static Mat regular(Mat mat, int length, int processType);
	
	/**
	 * 图像标准化，讲图像按照短边对其尺寸length，同时依照minStep和size切片
	 * 具体的逻辑是，如果图像长边l减去短边s的尺寸是(l-s) > (size-1)*minStep，则按照切片，平均分为size份
	 * (l-s) <= (size-1)*minStep 则 floor((l-s)/minStep)，一共切为 floor((l-s)/minStep)+1片
	 */
	static vector<Mat> regularSlice(Mat mat, int length, int size, int minStep);


private:
	/************************************************************************/
	/* 图像标准化，方块模式（长边方块，多余填补模式）                                                                   */
	/************************************************************************/
	static Mat regularSquare(Mat mat, int length);

	/************************************************************************/
	/* 图像标准化，原始模式                                                                     */
	/************************************************************************/
	static Mat regularOrigin(Mat mat, int length);

	/************************************************************************/
	/* 图像标准化，短边方块（长边方块，多余切除模式）                                                                     */
	/************************************************************************/
	static Mat regularSquareMidCut(Mat mat, int length);

}; 
