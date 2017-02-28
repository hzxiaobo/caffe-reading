#include "caffeclassifier.h"


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
Classifier::Classifier(int gpu_device, string proto_path, string model_path, string label_path, int crop_size, int r,
                       int g, int b) {

    std::cout << "status check: start initiating caffe classifier model " << std::endl; //系统开始运行时候打印一行表示进入到了初始化阶段（个人习惯而已，其实这里用cout不好）

    Caffe::set_mode(Caffe::GPU); //函数在common.cpp中，用来设置是GPU求解还是CPU求解，如果要使用CPU的话，请在这里使用Caffe::set_mode(Caffe::CPU);
    gpu_device_ = gpu_device;    //将全局device变量设为指定的device，后续求解的时候需要这个全局变量来指定device
    Caffe::SetDevice(gpu_device_); //函数在common.cpp中，设置proto以及模型load的gpu device（多gpu机器上需要指定gpu的device）

    net_ = new Net<float>(proto_path, TEST);    //函数在net.cpp里，用以load本地储存的proto.txt文件，然后初始化caffe网络
    //从训练好的模型中，拷贝参数
    net_->CopyTrainedLayersFrom(model_path); //函数在net.cpp中，从训练好的模型文件中，讲训练好的参数等文件拷贝到模型之中
    //注：值得注意的是，以上两个函数， 即 new Net<fload>(proto_path, TEST) 和 CopyTrainedLayersFrom(model_path)都是从从upgrade_proto.cpp里
    // 使用函数UpgradeNetAsNeeded 中读取本地文件，一个是读取file一个是读取二值file，将其转化为NetParameter之后，再load到模型中，
    // 而NetParameter是通过Google-protocol-buffers进行序列化的。so，读懂了NetParameter，就是明白了如何读取每一层的参数，以及初始化Net的

    Blob<float> *input_layer = net_->input_blobs()[0];     //修改输入层模板，他是通过调用的是net.cpp中相应的函数获取vector<Blob<Dtype>*> net_input_blobs_;它是网络初始化后的Blob数据，其中0是输入层的blob数据
    num_channels_ = input_layer->channels();    //获取blob中输入的图像的通道数
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);  //第一个参数是batch size的数量
    net_->Reshape();    /* Forward dimension change to all layers. */
    //以上5行函数是将net中的输入层，reshape下，事实上这件事情做的是强制的将input层的batch size resize到了1，然后再每一层都reshape下
    //通常情况下，input的的batch size 是由prototxt里控制的，这里因为是设置为单线程处理的，所以输入的batch size直接是设为1了，如果是以外层控制的方式输入的，则这里并不需要上面5行函数
    //当然，如果输入的batch size并不等于预设的情况下，也可以再函数中控制batch size的数量

    SetMean(crop_size, r, g, b); // 直接设定均值图像的大小与rgb均值，在googlenet之后，均值都是直接设定了直接相减的

    LoadTag(label_path);     //加载分类标签，原始的输出类别是数字，加个tag的话，看的清楚些

    std::cout << "status check: initiating caffe classifier model success " << std::endl;
}

/**
* 比较器
*/
static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
/* 返回数组v[] 最大值的前 N 个序号数组 */
static std::vector<int> Argmax(const std::vector<float> &v, int N) {
    std::vector <std::pair<float, int>> pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(),
                      PairCompare);
    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);

    pairs.clear();
    return result;
}

/* Return the top N predictions. 分类并返回最大的前 N 个预测 */
std::vector <Prediction> Classifier::Classify(const cv::Mat &img, int N) { //have been clean
    std::vector<float> output = Predict(img);
    std::cout << "c++ op: target probability of ";
    for (int i = 0; i < labels_.size(); i++) {
        std::cout << i << " " << output[i] << " , ";
    }
    std::cout << std::endl;
    std::vector<int> maxN = Argmax(output, N);
    std::vector <Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx])); //组成 [(标签，置信度),...]预测值数组
    }

    output.clear();
    maxN.clear();
    return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(string mean_char) { //have been clean
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_char, &blob_proto);
    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    // CHECK_EQ(mean_blob.channels(), num_channels_)
    //   << "Number of channels of mean file doesn't match input layer.";
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector <cv::Mat> channels;
    float *data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    channels.clear();
}

/**
* 直接设置文件的大小以及均值
*/
void Classifier::SetMean(int cropSize, float r, float g, float b) {
    std::vector <cv::Mat> channels;
    float *meanData;
    meanData = new float[cropSize * cropSize * 3];
    for (int i = 0; i < cropSize * cropSize; i++) {
        meanData[i] = r;
    }
    int startPo = cropSize * cropSize;
    for (int i = 0; i < cropSize * cropSize; i++) {
        meanData[startPo + i] = g;
    }
    startPo += cropSize * cropSize;
    for (int i = 0; i < cropSize * cropSize; i++) {
        meanData[startPo + i] = b;
    }

    float *data = meanData;

    for (int i = 0; i < num_channels_; ++i) {
        cv::Mat channel(cropSize, cropSize, CV_32FC1, data);
        channels.push_back(channel);
        data += cropSize * cropSize;
    }
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    channels.clear();
    delete[] meanData;
}

void Classifier::LoadTag(const string& label_path) {
    //	char *labelled_chars   = "/home/nisp/image-classify/c-working/models-porn/caffe_15_09_29_labeltag";
    const char *label_char = label_path.c_str();
    std::ifstream labels(label_char);
    string line;
    lableSize = 0;
    while (std::getline(labels, line)) {
        labels_.push_back(string(line));
        lableSize++;
    }
    std::cout << "labelSize is : " << lableSize << std::endl;
}

/*  分类 */
std::vector<float> Classifier::Predict(const cv::Mat &img) {
    //Blob<float>* input_layer = net_->input_blobs()[0];

    std::vector <cv::Mat> input_channels;
    WrapInputLayer(&input_channels);        //打包输入层
    Preprocess(img, &input_channels);       //数据预处理
    net_->ForwardPrefilled();               //前向计算
    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    //  const float* end = begin + output_layer->channels();
    const float *end =
            begin + lableSize; //训练数据时采用的是默认的1000类，所以output_layer->channels()是1000，这样是不对的，为了简单期间，色情图像只是分为2类，所以使用这个2，以后再改

    input_channels.clear();
    //input_layer = NULL;
    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer.
打包网络中不同的的输入层 cv:Mat 对象
（每个通道一个）。这样我们保存一个 memcpy的操作，我们
并不需要依靠cudaMemcpy2D 。最后预处理
操作将直接写入不同通道的输入层。
*/
void Classifier::WrapInputLayer(std::vector <cv::Mat> *input_channels) {    //have been clear

    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
    input_layer = NULL;
    input_data = NULL;
}

//数据预处理
void Classifier::Preprocess(const cv::Mat &img,
                            std::vector <cv::Mat> *input_channels) {            //all have been clear
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    //通道数据根据设置进行转换
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);     // 三通道(彩色)
    else
        sample_resized.convertTo(sample_float, CV_32FC1);     // 单通道    (灰度)

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels.
    此操作将数据 BGR 直接写入输入层对象input_channels */
    cv::split(sample_normalized, *input_channels);

}

/**
* load本地图像，判断是否时porn图像
*/
int Classifier::isTarget(const cv::Mat &img) {
    try {
        if (img.cols == 0 || img.rows == 0) {
            std::cout << "image load failed" << std::endl;
            return 0;
        }
        //以下注释的部分是图像是整体一块输入的
        //				vector < Mat > images = imgBaseProcess::regularSlice(img, 256, 3, 30); //have been clear
        //				std::cout << "@slices size is : " << images.size() << std::endl;
        int pornTag = -1;
        //				for (int i = 0; i < images.size(); i++) {
        std::vector <Prediction> predictions = Classify(img, labels_.size()); //分类 have been clear
        /* Print the top N predictions. 打印前N 个预测值*/

        string tag = hardFilter(predictions, "normal", 0.33);


        predictions.clear();
        for (int i = 0; i < labels_.size(); i++) {
            if (tag == labels_[i]) {
                return i;
            }
        }
        return 0;
        //		if (tag == "normal") {
        //			return 0;
        //		} else if (tag == "terror") {
        //			return 1;
        //		} else {
        //			return 0;
        //		}

    } catch (Exception e) {
        return -1;
    }
}

/**
* 设置图像的大小
*/
void Classifier::setResizeSize(int size) {
    resizeSize = size;
}

/**
* 检查输入的图像流是否是色情图像
*/
char *Classifier::targetCheck(char *ipArr, int ipArrLength) {
    try {
        if (ipArrLength < 1) {
            return 0;
        }
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_device_);

        vector<char> imgArr(ipArrLength); //have been clear
        std::cout << ">>>device id is : " << gpu_device_ << " & the length of image is : " << imgArr.size()
                  << " classifer is : " << class_id << std::endl;
        char *mpDst = &imgArr[0]; //已指向null
        memcpy(mpDst, ipArr, ipArrLength);
        Mat img = imdecode(imgArr, -1);
        imgArr.clear();
        mpDst = NULL;

        if (img.cols < 4 || img.rows < 4) {
            std::cout << ">>>image load failed" << std::endl;
            return 0;
        }

        std::vector <Prediction> predictions = Classify(img, labels_.size()); //分类 have been clear

        string opResult = "";

        for (int i = 0; i < labels_.size(); i++) {
            opResult.append(predictions[i].first);
            opResult.append(":");
            int tmp = floor(predictions[i].second * 100.0);
            opResult.append(baseform_int2Str(tmp));
            if (i != (labels_.size() - 1)) {
                opResult.append(",");
            }
        }
        std::cout << "c++ op: target opResult : " << opResult << std::endl;

//	        string mResultStr = "-1";
        char *ch = (char *) malloc(sizeof(char) * (opResult.length() + 1));
        strcpy(ch, opResult.c_str());
        return ch;


//			string tag = hardFilter(predictions, "normal", 0.33);
//			predictions.clear();
//			for(int i = 0 ; i < labels_.size() ;i++){
//				if (tag == labels_[i]){
//					return i;
//				}
//			}
//			return 0;

    } catch (Exception e) {
        std::cout << ">>>c++ error" << std::endl;
    }

    string mResultStr = "-1";
    char *ch = (char *) malloc(sizeof(char) * (mResultStr.length() + 1));
    strcpy(ch, mResultStr.c_str());
    return ch;
}


/**
* 对于预测再过滤，主要是为了处理某些normal 0.6，porn 0.3的情况，这种情况线上使用的时候应当一律归结为terror
*/
string Classifier::hardFilter(std::vector <Prediction> predictions,
                              string normalTag, float ratio) {
    if (predictions[0].first != normalTag) {
        return predictions[0].first;
    } else {
        float normalPro = predictions[0].second;
        float carePro = 1 - normalPro;
        if (normalPro == 0) {
            if (predictions.size() <= 1) {
                return "care";
            } else {
                return predictions[1].first;
            }
        }
        if (carePro / normalPro > ratio) {
            if (predictions.size() <= 1) {
                return "care";
            } else {
                return predictions[1].first;
            }
        } else {
            return predictions[0].first;
        }
    }
}

shared_ptr <Blob<float>> Classifier::GetLayerOutput(const cv::Mat &img, string fc) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_device_);

    std::vector <cv::Mat> input_channels;
    WrapInputLayer(&input_channels);        //打包输入层
    Preprocess(img, &input_channels);       //数据预处理
    net_->ForwardPrefilled();               //前向计算

    shared_ptr <Blob<float>> outputBlob = net_->blob_by_name(fc);
    return outputBlob;
}


shared_ptr <Blob<float>> Classifier::GetBlobByName(string blob_name) {
    shared_ptr <Blob<float>> outputBlob = net_->blob_by_name(blob_name);
    return outputBlob;
}


void Classifier::Forward(const cv::Mat &img) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_device_);
    std::vector <cv::Mat> input_channels;
    WrapInputLayer(&input_channels);        //打包输入层
    Preprocess(img, &input_channels);       //数据预处理
    net_->ForwardPrefilled();               //前向计算
}


string Classifier::CheckTarget(const cv::Mat &img) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_device_);

    std::vector <cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    net_->ForwardPrefilled();

    std::vector <Prediction> predictions = Classify(img, labels_.size()); //分类 have been clear

    string opResult = "";
    for (int i = 0; i < labels_.size(); i++) {
        opResult.append(predictions[i].first);
        opResult.append(":");
        int tmp = floor(predictions[i].second * 100.0);
        opResult.append(baseform_int2Str(tmp));
        if (i != (labels_.size() - 1)) {
            opResult.append(",");
        }
    }
    std::cout << "c++ op: target opResult : " << opResult << std::endl;

    return opResult;
}

