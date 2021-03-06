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

/**
 * Pair (label, confidence) representing a prediction.
 * pair(标签，置信度)  预测值
 */
typedef std::pair<string, float> Prediction;

/**
 * Return the indices of the top N values of vector v.
 * 返回数组v[] 最大值的前 N 个序号数组
 * @param v 输入的无序的数组v
 * @param N 输出最大的前N个数字的排序
 * @return 以从大到小的顺序返回输入序列中的前N个数在序列中的位置
 */
static std::vector<int> Argmax(const std::vector<float> &v, int N);


/**
 * 比较器，比较两个pair里的数值哪个大，用来选取较大的那个pair数值
 * 使用的地点在Argmax中，是partial_sort中的比较器
 * @param lhs 待比较的pair 1
 * @param rhs 待比较的pair 2
 * @return 如果lhs>rhs 输出true
 */
static bool PairCompare(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs);


/**
 * 本代码是README.md中的第一种解决caffe forward的多线程的解决方案，即
    1.修改调用方，使其load多个Classifier类，即初始化多个Net和Blob，然后在调用方控制多线程的数量；
    （当然这个是最低效的，GPU根本跑不满，然而却是实现成本最低的）
 **/
/*  基于caffe的分类接口类 Classifier */
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
     * 输入一张图像的Mat，输出识别的结果，用数字来代替类别，通常是用在本地调用
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

    /**
     * 输入一张图像，并获取指定层的输出，较为常用的情况是去softmax前的一层的分类结果层的输出
     * 这一层的输出通常情况下是属于某一类的得分，可以让外层调用者通过参数来获得高召回，还是高精确度的结果
     * @param img 输入的Mat图像
     * @param fc 指定的输出层的名字
     * @return 返回指定输出层的各个类别的结果
     */
    shared_ptr <Blob<float> > GetLayerOutput(const cv::Mat &img, string fc);

    /**
     * 获取某一层的结果，与GetLayerOutput的结果不同，上面的函数的输入是Mat和指定层的名字
     * 而本函数只是输出指定层的结果，他与下面的Forward函数合起来，才是上面的GetLayerOutput函数
     * @param blob_name
     * @return
     */
    shared_ptr <Blob<float> > GetBlobByName(string blob_name);

    /**
     * 输入一张图像，并进行后向传递，本函数与GetBlobByName合起来才是GetLayerOutput这个函数
     * @param img 输入的图像
     */
    void Forward(const cv::Mat &img);

    /**
     * 输入一张图像的Mat，并返回分类的结果，会将所有类别以及类别的score输出，是softmax之后的结果×100之后
     * 通过指定的排列顺序进行输出的
     * @param img
     * @return 返回类别+概率，需要调用端依据格式解析,其格式是：  类别1:score1,类别2:score2,...， 其中score*均为 100分制的
     */
    string CheckTarget(const cv::Mat &img);

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
     * 里面实现了两种方式，一个是caffe原生的使用opencv指针的方式，另外一个则是使用memcpy的方式（已暂时注释掉了，需要使用的话，可以启用）
     * @param img 待预测的图像
     * @return 返回所有类别的概率
     */
    std::vector<float> Predict(const cv::Mat &img);

    /**
     * Wrap the input layer of the network in separate cv::Mat objects
     * (one per channel). This way we save one memcpy operation and we
     * don't need to rely on cudaMemcpy2D. The last preprocessing
     * operation will write the separate channels directly to the input layer.
     * 打包网络中不同的的输入层 cv:Mat 对象（每个通道一个）。这样我们保存一个 memcpy的操作，我们
     * 并不需要依靠cudaMemcpy2D 。最后预处理操作将直接写入不同通道的输入层。
     * 使用opencv指针的方式，将图像的Mat数据和Blob的数据放在共同的指针地址
     * 值得注意的是，调用这个函数的时候，并不需要input_channels里有要处理的图像数据
     * 这个是caffe源码里自带的原生的调用方式
     * @param input_channels 要使用指针将Blob与Mat<vector>变成共同指针的输入数据
     */
    void WrapInputLayer(std::vector <cv::Mat> *input_channels);

    /**
     * 通过使用memcpy的方式，将输入打包拷贝到input_layer层
     * 这个函数的使用功能等同于WrapInputLayer，只不过使用顺序方面，input_channels里必须是有要处理的图像数据才行
     * 其正确的调用方式是在Predict(const cv::Mat &img)里的way second（已经被注释掉了，需要看的话，可以使用这个方式
     * @param input_channels 已经填入拆了通道的图像的数据
     */
    void PushInputLayer(std::vector <cv::Mat> *input_channels);

    /**
     * 数据预处理，将输入的图像经过 通道分离，图像resize，以及减去均值后，将其压入相应的input_channels中
     * @param img 输入的图像
     * @param input_channels 相应的input_channels
     */
    void Preprocess(const cv::Mat &img, std::vector <cv::Mat> *input_channels);

    /**
     * 输出指定层的数据，用来检查函数调用以及输出是否正确
     * @param layer_name 待查看数据的layer层的名字
     * @param po 从第po个开始输出
     * @param n 从po处开始向后数n个位置，都展示出来
     */
    void ShowLayerData(string layer_name, int po = 0, int n = 5);

private:
    Net<float> *net_;                               //caffe分类网络对象
    cv::Size input_geometry_;                       //输入图像几何尺寸
    int num_channels_;                              //网络通道数
    cv::Mat mean_;                                  //均值图像，
    std::vector <string> labels_;                   //目标标签数组
    int label_size_;                                //类别的数目，用在截断类别输出时使用，如果是使用别人的模型直接finetuning的时候，类别是原始的类别，所以需要截断
    int gpu_device_;                                //设备id，在多显卡的服务器上，用来指定某个显卡来进行计算
};
