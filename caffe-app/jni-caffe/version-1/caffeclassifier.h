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

/* Return the indices of the top N values of vector v. */
/* 返回数组v[] 最大值的前 N 个序号数组 */
static std::vector<int> Argmax(const std::vector<float> &v, int N);

/**
 * 比较器，比较两个pair里的数值哪个大，用来选取较大的那个pair数值
 * 使用的地点在Argmax中，是partial_sort中的比较器
*/
static bool PairCompare(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs);


/**
 * 本代码是README.md中的第一种解决caffe forward的多线程的解决方案，即
    1.修改调用方，使其load多个Classifier类，即初始化多个Net和Blob，然后在调用方控制多线程的数量；
    （当然这个是最低效的，GPU根本跑不满，然而却是实现成本最低的）
 **/
/*  分类接口类 Classifier */
class Classifier {
public:

    /**
     * 类的初始化构造函数，首先load prototxt中的深度学习的网络模型，随后load 训练好的参数模型，然后再将网络形状，均值等均设置好，相当于初始化阶段
     * @param gpu_device 多显卡机器上需要指定显卡id
     * @param proto_path 深度学习模型xx.prototxt的本地路径
     * @param model_path 训练好的模型参数的本地路径
     * @param label_path 输出的label的本地路径，实际上这个输出的label原本不需要放在c++端，放在调用方的那一端即可
     * @param crop_size 实际上是输入图像的大小（在调用方resize之后传入这里）
     * @param r 图像的平均值，r通道
     * @param g 图像的平均值，g通道
     * @param b 图像的平均值，b通道
     * @return 类的初始化构造函数，所以没有return的值
     */
    Classifier(int gpu_device, string proto_path, string model_path, string label_path, int r, int g, int b);

    /**
     * Return the top N predictions. 分类并返回最大的前 N 个预测，是调用Predict函数完成分类，然后再在本函数里完成筛选前N个概率最大的类别的工作
     * @param img 输入的Mat图像
     * @param 前N个预测，默认输入的数字是1
     * @return 返回预测结果，以vector的形式存放，（Prediction的定义详见本.h文件里的Prediction
     */
    std::vector <Prediction> Classify(const cv::Mat &img, int N = 1); //分类，默认返回前5个预测值[(标签，置信度),... ] 数组

    /**
     * 输入一张图像的Mat，输出识别的结果，用数字来代替类别
     * @param img 输入的图像
     * @return 输出是否是待检测的目标（如果是2类分类的情况，通常情况下，1代表是目标，0代表不是目标
     */
    int isTarget(const cv::Mat &img);

    /**
     * 输入一张图像的流数据，输出分类的结果（含分类概率），其结果是封在了char *中，输出到调用端，供调用端解析
     * 当然，更为合适的方式是直接拿到blob层的结果，直接将该层的结果输出出去也就是下面
     * @param ipArr 输入的图像流
     * @param ipArrLength 输入的图像流的长度
     * @return 返回类别+概率，需要调用端依据格式解析,其格式是：  类别1:score1,类别2:score2,...， 其中score*均为 100分制的
     */
    char *targetCheck(char *ipArr, int ipArrLength);

    float *targetFeatureExtract(char *ipArr, int ipArrLength, string fc, int feature_length);

    float *showFeatureExtract(const cv::Mat &img, string fc, int feature_length);

    shared_ptr <Blob<float> > GetLayerOutput(const cv::Mat &img, string fc);

    shared_ptr <Blob<float> > GetBlobByName(string blob_name);

    void Forward(const cv::Mat &img);

    string CheckTarget(const cv::Mat &img);

//    void setResizeSize(int size);

private:

    /**
     * 值得注意的是，这个函数里的设置实在是太为繁琐了，讲道理应该是有优化的空间的，可以直接省去绝大部分的代码
     * 函数直接设定均值图像的r，g，b通道的均值, GoogLeNet及其之后，均值都是直接设定了直接相减的
     * @param r r通道的均值
     * @param g g通道的均值
     * @param b b通道的均值
     */
    void SetMean(float r, float g, float b);

    /**
     * 加载分类标签，原始的输出类别是数字，加个tag的话，类别输出明显些,tag的存放格式很简单，一行就是一个tag
     * @param label_path 存放tag标签的路径，
     */
    void LoadTag(const string& label_path);

    /**
     * 预测一张图像的分类，以vector<float>的格式输出，
     * @param img 待预测的图像
     * @return 返回所有类别的概率
     */
    std::vector<float> Predict(const cv::Mat &img);

    void WrapInputLayer(std::vector <cv::Mat> *input_channels);

    /**
     * 数据预处理，将输入的图像经过 通道分离，图像resize，以及减去均值后，将其压入相应的input_channels中
     * @param img 输入的图像
     * @param input_channels 相应的input_channels
     */
    void Preprocess(const cv::Mat &img, std::vector <cv::Mat> *input_channels);

    float *ExtractFeature(const cv::Mat &img, string fc);

private:
    Net<float> *net_;                           //caffe分类网络对象
    cv::Size input_geometry_;                   //输入图像几何尺寸
    int num_channels_;                          //网络通道数
    cv::Mat mean_;                              //均值图像，
    std::vector <string> labels_;                //目标标签数组
//    int resizeSize;
    int label_size_;                                //类别的数目，用在截断类别输出时使用，如果是使用别人的模型直接finetuning的时候，类别是原始的类别，所以需要截断
    int gpu_device_;                                //设备id
};
