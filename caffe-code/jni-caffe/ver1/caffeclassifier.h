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
 * 本代码是README.md中的第一种解决caffe forward的多线程的解决方案，即
    1.修改调用方，使其load多个Classifier类，即初始化多个Net和Blob，然后在调用方控制多线程的数量；
    （当然这个是最低效的，GPU根本跑不满，然而却是实现成本最低的）
 **/
/*  分类接口类 Classifier */
class Classifier {
public:

    /**
      * 类的初始化构造函数，各个参数的意义分别是：
      * int: int gpu_device， 多显卡机器上需要指定显卡id / string proto_path: 深度学习模型的 xx.prototxt的本地路径 / string model_path:训练好的模型参数的本地路径
      * string label_path: 输出的label的本地路径，实际上这个输出的label原本不需要放在c++端，放在调用方的那一端即可
      * int crop_size，实际上是输入图像的大小（在调用方resize之后传入这里）
      * int r, int g, int b, 图像的平均值，在GoogLeNet之后基本上都是直接减去平均值的，而在早期的AlexNet的时候，是向模型中传入平均值的一张图像的
      *
      * 函数作用：
      * 首先load prototxt中的深度学习的网络模型，随后load 训练好的参数模型，然后再将网络形状，均值等均设置好，相当于初始化阶段
    **/
    Classifier(int gpu_device, string proto_path, string model_path, string label_path, int crop_size, int r,
               int g, int b);

    std::vector <Prediction> Classify(const cv::Mat &img, int N = 1); //分类，默认返回前5个预测值[(标签，置信度),... ] 数组

    int isTarget(const cv::Mat &img);

    char *targetCheck(char *ipArr, int ipArrLength);

    float *targetFeatureExtract(char *ipArr, int ipArrLength, string fc, int feature_length);

    float *showFeatureExtract(const cv::Mat &img, string fc, int feature_length);

    shared_ptr <Blob<float>> GetLayerOutput(const cv::Mat &img, string fc);

    shared_ptr <Blob<float>> GetBlobByName(string blob_name);

    void Forward(const cv::Mat &img);

    string CheckTarget(const cv::Mat &img);

    void setResizeSize(int size);

private:
    void SetMean(string mean_char); //load mean file

    void SetMean(int cropSize, float r, float g, float b); //set mean rgb like googlenet use

    void LoadTag(const char *labelled_char); //load labelled char file

    std::vector<float> Predict(const cv::Mat &img);                    //预测

    void WrapInputLayer(std::vector <cv::Mat> *input_channels);

    void Preprocess(const cv::Mat &img, std::vector <cv::Mat> *input_channels);

    string hardFilter(std::vector <Prediction> predictions, string normalTag, float ratio);

    float *ExtractFeature(const cv::Mat &img, string fc);

private:
    Net<float> *net_;                        //caffe分类网络对象
    cv::Size input_geometry_;                   //输入图像几何尺寸
    int num_channels_;                            //网络通道数
    cv::Mat mean_;                              //均值图像
    std::vector <string> labels_;                //目标标签数组
    int resizeSize;
    int lableSize;                                //类别的数目
    int gpu_device_;                                //设备id
};
