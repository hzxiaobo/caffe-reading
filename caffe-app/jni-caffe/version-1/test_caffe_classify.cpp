#include <iostream>
#include "highgui.h"
#include "cv.h"
#include "caffeclassifier.h"
#include "imgBaseProcess.h"
#include "fileprocess.h"
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Classifier* classifier = new Classifier(0,"/home/xiaobo/work/caffe-test/new-version/target/models/terror.prototxt","/home/xiaobo/work/caffe-test/new-version/target/models/terror.caffemodel","/home/xiaobo/work/caffe-test/new-version/target/models/terror.labeltag",
	104,117,123); //创建分类器
	string listFile = "../samples/testList.list";
	string loadPath = "../samples/";
	vector < string > fileList = FileProcess::readFileList(listFile);
	for (int i = 0; i < fileList.size(); i++) {
		std::cout << "---------- Prediction for " << loadPath << fileList[i]
				<< " ----------" << std::endl;
		cv::Mat img = cv::imread(loadPath + fileList[i], -1);

		clock_t a = clock();
		int result = classifier->isTarget(img);
		cout << "result is : " << result << endl;
		clock_t b = clock();
		cout << "total time is : " << (double) (b - a) / CLOCKS_PER_SEC << endl;
	}
}
