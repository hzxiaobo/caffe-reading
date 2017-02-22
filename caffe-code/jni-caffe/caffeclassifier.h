#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "cv.h"
#include "highgui.h"
#include "imgBaseProcess.h"
#include "base_form.h"

using namespace caffe;
// NOLINT(build/namespaces)
using namespace cv;

/* Pair (label, confidence) representing a prediction. */
/* pair(标签，置信度)  预测值 */
typedef std::pair<string, float> Prediction;


/**
 * 值得注意的是，本类是使用caffe原生的前向传递的方式进行求解的，因此本类不支持多线程求解
 * 如果要达到多线程求解，方案通常有4个
 * 1.在调用方下功夫，即load多个Classifier类，即初始化多个Net和Blob，然后在调用方控制多线程的数量（当然这个是最低效的，然而却是实现成本最低的）
 * 2.同样是在调用方下功夫，但是在传入求解的图像的时，不是使用单张图像，而是使用图像batch的方式，这种方式会稍微麻烦些，但是却能达到高效
 * 3.针对特定的网络直接写一份代码，将每一个layer写成一个函数，每个layer的参数独自读取，通过函数的方式实现多线程，这样运行效率较高但是却会导致书写的坑比较多，容易出错
 * 4.写个通用的网络传递的代码（求大神指点，没想到怎么写，多进程？）
 **/
/*  分类接口类 Classifier */
class Classifier {
public:

    /**
      * 类的初始化构造函数，各个参数的意义分别是：
      * int: gpu device id， 多显卡机器上需要指定显卡id / string model_file: 深度学习模型的 xx.prototxt的本地路径 / string trained_file:训练好的模型参数的本地路径
      * string labelled_file: 输出的label的本地路径，实际上这个输出的label原本不需要放在c++端，放在调用方的那一端即可
      * int cropSize，实际上是输入图像的大小（在调用方resize之后传入这里）
      * int r, int g, int b, 图像的平均值，在GoogLeNet之后基本上都是直接减去平均值的，而在早期的AlexNet的时候，是向模型中传入平均值的一张图像的
      *
      * 函数作用：
      * 首先load prototxt中的深度学习的网络模型，随后load 训练好的参数模型，然后再将网络形状，均值等均设置好，相当于初始化阶段
    **/
	Classifier(int de_id, string model_file, string trained_file, string labelled_file, int cropSize, int r, int g, int b);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 1); //分类，默认返回前5个预测值[(标签，置信度),... ] 数组

	int isTarget(const cv::Mat& img);

	char* targetCheck(char* ipArr, int ipArrLength);

	float* targetFeatureExtract(char* ipArr, int ipArrLength, string fc, int feature_length);

	float* showFeatureExtract(const cv::Mat& img, string fc, int feature_length);

    shared_ptr<Blob<float> > GetLayerOutput(const cv::Mat& img, string fc);

    shared_ptr<Blob<float> > GetBlobByName(string blob_name);

    void Forward(const cv::Mat& img);

    string CheckTarget(const cv::Mat& img);

	void setResizeSize(int size);

private:
	void SetMean(string mean_char); //load mean file

	void SetMean(int cropSize, float r, float g, float b); //set mean rgb like googlenet use

	void LoadTag(const char* labelled_char); //load labelled char file

	std::vector<float> Predict(const cv::Mat& img);                    //预测

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

	string hardFilter(std::vector < Prediction > predictions, string normalTag , float ratio);

	float* ExtractFeature(const cv::Mat& img, string fc) ;

private:
	Net<float> *net_;               			//caffe分类网络对象
	cv::Size input_geometry_;                   //输入图像几何尺寸
	int num_channels_;                         	//网络通道数
	cv::Mat mean_;                              //均值图像
	std::vector<string> labels_;                //目标标签数组
	int resizeSize;
	int lableSize;								//类别的数目
	int device_id;								//设备id
};
