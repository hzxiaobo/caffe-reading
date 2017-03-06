#include "caffeclassifier.h"


Classifier::Classifier(int gpu_device, string proto_path, string model_path, string label_path, int r, int g, int b) {

    std::cout << "status check: start initiating caffe classifier model "
              << std::endl; //系统开始运行时候打印一行表示进入到了初始化阶段（个人习惯而已，其实这里用cout不好）
    //测试用，用来设置为CPU模式
    Caffe::set_mode(Caffe::CPU); //函数在common.cpp中，用来设置是GPU求解还是CPU求解，如果要使用CPU的话，请在这里使用Caffe::set_mode(Caffe::CPU);
    gpu_device_ = -1;    //将全局device变量设为指定的device，后续求解的时候需要这个全局变量来指定device

//    Caffe::set_mode(Caffe::GPU); //函数在common.cpp中，用来设置是GPU求解还是CPU求解，如果要使用CPU的话，请在这里使用Caffe::set_mode(Caffe::CPU);
//    gpu_device_ = gpu_device;    //将全局device变量设为指定的device，后续求解的时候需要这个全局变量来指定device
//    Caffe::SetDevice(gpu_device_); //函数在common.cpp中，设置proto以及模型load的gpu device（多gpu机器上需要指定gpu的device）

    net_ = new Net<float>(proto_path, TEST);    //函数在net.cpp里，用以load本地储存的proto.txt文件，然后初始化caffe网络
    //从训练好的模型中，拷贝参数
    net_->CopyTrainedLayersFrom(model_path); //函数在net.cpp中，从训练好的模型文件中，讲训练好的参数等文件拷贝到模型之中
    //注：值得注意的是，以上两个函数， 即 new Net<fload>(proto_path, TEST) 和 CopyTrainedLayersFrom(model_path)都是从从upgrade_proto.cpp里
    // 使用函数UpgradeNetAsNeeded 中读取本地文件，一个是读取file一个是读取二值file，将其转化为NetParameter之后，再load到模型中，
    // 而NetParameter是通过Google-protocol-buffers进行序列化的。so，读懂了NetParameter，就是明白了如何读取每一层的参数，以及初始化Net的

    Blob<float> *input_layer = net_->input_blobs()[0];     //修改输入层模板，他是通过调用的是net.cpp中相应的函数获取vector<Blob<Dtype>*> net_input_blobs_;它是网络初始化后的Blob数据，其中0是输入层的blob数据
    num_channels_ = input_layer->channels();    //获取blob中输入的图像的通道数
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height()); //设置输入图像的geometry尺寸
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);  //第一个参数是batch size的数量
    net_->Reshape();    /* Forward dimension change to all layers. */
    //以上5行函数是将net中的输入层，reshape下，事实上这件事情做的是强制的将input层的batch size resize到了1，然后再每一层都reshape下
    //通常情况下，input的的batch size 是由prototxt里控制的，这里因为是设置为单线程处理的，所以输入的batch size直接是设为1了，如果是以外层控制的方式输入的，则这里并不需要上面5行函数
    //当然，如果输入的batch size并不等于预设的情况下，也可以再函数中控制batch size的数量

    SetMean(r, g, b); // 直接设定均值图像的大小与rgb均值，GoogLeNet及其之后，均值都是直接设定了直接相减的

    LoadTag(label_path);     //加载分类标签，原始的输出类别是数字，加个tag的话，类别输出明显些

    std::cout << "status check: initiating caffe classifier model success " << std::endl;
}


std::vector <Prediction> Classifier::Classify(const cv::Mat &img, int N) {
    std::vector<float> output = Predict(img);   //使用预测函数进行预测
    std::vector<int> maxN = Argmax(output, N);  //将输出的结果排序，取前N个最大的值
    std::vector <Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx])); //组成 [(标签，置信度),...]预测值数组
    }
    output.clear();
    maxN.clear();
    return predictions;
}


void Classifier::SetMean(float r, float g, float b) {
    mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(r,g,b));
//    std::cout << "param check @ SetMean, values of (1,1) :" << mean_.at<Vec3f>(1,1)[0] << " vs " << mean_.at<Vec3f>(1,1)[1]  << " vs " << mean_.at<Vec3f>(1,1)[2]<< std::endl;
//    getchar();
}

void Classifier::LoadTag(const string &label_path) {
    const char *label_char = label_path.c_str();
    std::ifstream labels(label_char);       //读取label_path所指定的文件
    string line;                            //缓存
    label_size_ = 0;
    while (std::getline(labels, line)) {
        labels_.push_back(string(line));
        label_size_++;
    }
}


std::vector<float> Classifier::Predict(const cv::Mat &img) {
    //Blob<float>* input_layer = net_->input_blobs()[0];

    std::vector <cv::Mat> input_channels;
    Preprocess(img, &input_channels);       //数据预处理
    PushInputLayer(&input_channels);        //打包输入层


    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for(int i = 0 ; i < width*height ; i++){
        if (i < 10 || i >= width*height - 10){
            std::cout << *input_data << " " ;
        }
        input_data ++ ;
    }
    std::cout << std::endl;
    for(int i = 0 ; i < width*height ; i++){
        if (i < 10 || i >= width*height - 10){
            std::cout << *input_data << " " ;
        }
        input_data ++ ;
    }
    std::cout << std::endl;
    for(int i = 0 ; i < width*height ; i++){
        if (i < 10 || i >= width*height - 10){
            std::cout << *input_data << " " ;
        }
        input_data ++ ;
    }
    std::cout << std::endl;

    std::cout << " gl input next showdata type " << std::endl;

    ShowLayerData("input_layer", 0, 10);
    ShowLayerData("input_layer", 224*224 - 10, 10);
    ShowLayerData("input_layer", 224*224, 10);
    ShowLayerData("input_layer", 224*224*2 - 10, 10);
    ShowLayerData("input_layer", 224*224*2 , 10);
    ShowLayerData("input_layer", 224*224*3 - 10, 10);

    std::cout << std::endl;



    net_->ForwardPrefilled();               //前向计算
    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + label_size_;
    //训练数据时采用的是默认的1000类，所以用原始的const float* end = begin + output_layer->channels()是不对的，
    // 因为output_layer->channels()是1000，因为输入的类别数目并不等于1000，是label_size_
    input_channels.clear();
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
void Classifier::PushInputLayer(std::vector <cv::Mat> *input_channels) {

    Blob<float> *input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();

    memcpy(input_data, ((*input_channels)[0].data), width*height*4);
    input_data += width*height;
    memcpy(input_data, ((*input_channels)[1].data), width*height*4);
    input_data += width*height;
    memcpy(input_data, ((*input_channels)[2].data), width*height*4);


//    vector<char> imgArr(ipArrLength);
//    std::cout << "param check, device id is : " << gpu_device_ << " & the length of image is : " << imgArr.size() << std::endl;
//    char *mpDst = &imgArr[0]; //已指向null
//    memcpy(mpDst, ipArr, ipArrLength);
//
//    memcpy()
//    input_channels[0].data;
//
//    for (int i = 0; i < input_layer->channels(); ++i) {
//        cv::Mat channel(height, width, CV_32FC1, input_data);
//        input_channels->push_back(channel);
//        input_data += width * height;
//    }
//    input_layer = NULL;
//    input_data = NULL;
}



void Classifier::WrapInputLayer(std::vector <cv::Mat> *input_channels) {

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
        sample_resized.convertTo(sample_float, CV_32FC1);     // 单通道(灰度)

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels.
    此操作将数据 BGR 直接写入输入层对象input_channels */
    cv::split(sample_normalized, *input_channels);
}


/**
 * 实现的有些低效，并不需要这样来实现，如果要输出数字，其实不需要转一道Prediction的
 */
int Classifier::isTarget(const cv::Mat &img) {
    try {
        if (img.cols == 0 || img.rows == 0) {
            std::cout << "image load failed" << std::endl;
            return 0;
        }
        std::vector <Prediction> predictions = Classify(img, labels_.size());
        string tag = predictions[0].first;
        predictions.clear();
        for (int i = 0; i < labels_.size(); i++) {
            if (tag == labels_[i]) {
                return i;
            }
        }
        return 0;
    } catch (Exception e) {
        return -1;
    }
}


char *Classifier::targetCheck(char *ipArr, int ipArrLength) {
    try {
        if (ipArrLength < 1) {
            return 0;
        }
        //值得注意的是下面两句话，当初始化初始的是GPU的时候，下面这两句话并不起作用
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_device_);

        //下面这段opencv的代码是直接将图像处理的byte[]从内存里直接转为Mat，而不需要将图像储存在本地，节省两次io（储存一次，删除一次）
        vector<char> imgArr(ipArrLength);
        std::cout << "param check, device id is : " << gpu_device_ << " & the length of image is : " << imgArr.size() << std::endl;
        char *mpDst = &imgArr[0]; //已指向null
        memcpy(mpDst, ipArr, ipArrLength);
        Mat img = imdecode(imgArr, -1);//imdecode里的int flags跟imread()里的int flags意义一样，1代表3通道，0代表灰度图，-1代表图是什么通道的就load什么通道
        imgArr.clear();
        mpDst = NULL;

        //检查下图像的尺寸，看看是否转化成功
        if (img.cols == 0 || img.rows == 0) {
            std::cout << "warn, load image failed!" << std::endl;
            return 0;
        }

        std::vector <Prediction> predictions = Classify(img, labels_.size());

        //以下是拼接从c++端到java端的输出
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
        std::cout << "result check, op result is: " << opResult << std::endl;

        //注意下面这段代码，之所以采用的malloc的方式，是因为直接将char*或者string 输出的时候，会有小概率输出为空（大概10%~20%左右），所以只能这样输出
        char *ch = (char *) malloc(sizeof(char) * (opResult.length() + 1));
        strcpy(ch, opResult.c_str());
        return ch;

    } catch (Exception e) {
        std::cout << "error, c++ process error" << std::endl;
    }
    //如果出现了error，就直接返回一个-1给到外面的java端好了
    string mResultStr = "-1";
    char *ch = (char *) malloc(sizeof(char) * (mResultStr.length() + 1));
    strcpy(ch, mResultStr.c_str());
    return ch;
}


shared_ptr <Blob<float> > Classifier::GetLayerOutput(const cv::Mat &img, string fc) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_device_);

    std::vector <cv::Mat> input_channels;
    WrapInputLayer(&input_channels);        //打包输入层
    Preprocess(img, &input_channels);       //数据预处理
    net_->ForwardPrefilled();               //前向计算

    shared_ptr <Blob<float> > outputBlob = net_->blob_by_name(fc);
    return outputBlob;
}


shared_ptr <Blob<float> > Classifier::GetBlobByName(string blob_name) {
    shared_ptr <Blob<float> > outputBlob = net_->blob_by_name(blob_name);
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
    std::cout << "result check: target opResult : " << opResult << std::endl;

    return opResult;
}


static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
    return lhs.first > rhs.first;
}


static std::vector<int> Argmax(const std::vector<float> &v, int N) {
    std::vector <std::pair<float, int> > pairs;
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


void Classifier::ShowLayerData(string layer_name, int po, int n){
    if (n <= 0 ){ //如果n的数目小于等于0 则直接return
        return ;
    }
    //以下这一部分待测试，需要在caffe环境下测试可用性
    Blob<float> *show_layer;
    if (layer_name.compare("input_layer") == 0){
        show_layer = net_->input_blobs()[0];
    }
//    else {
//        shared_ptr <Blob<float>> show_layer = net_->blob_by_name(layer_name);
//    }

    float *show_data = show_layer->mutable_cpu_data();
    show_data += po;
    for(int i = 0 ; i < n ; i++){
        std::cout << *show_data << " " ;
        show_data ++ ;
    }
    std::cout << std::endl;

}
