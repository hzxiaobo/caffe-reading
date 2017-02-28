//
// Created by hzzhangxiaobo on 2017/2/28.
//

//从jni-caffe, ver1里摘出来的代码，这部分代码是使用blobProto读取均值文件，并将网络的均值设置出来的代码
//直接解除注释就可以贴到相应的代码里使用，如果有需要的话
//void SetMean(string mean_char);  //加载储存均值的file，在AlexNet时代，通常是由这个函数来load均值文件来着
//
///* Load the mean file in binaryproto format. */
//void Classifier::SetMean(string mean_char) {
//    BlobProto blob_proto;
//    ReadProtoFromBinaryFileOrDie(mean_char, &blob_proto);
//    /* Convert from BlobProto to Blob<float> */
//    Blob<float> mean_blob;
//    mean_blob.FromProto(blob_proto);
//    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
//    std::vector <cv::Mat> channels;
//    float *data = mean_blob.mutable_cpu_data();
//    for (int i = 0; i < num_channels_; ++i) {
//        /* Extract an individual channel. */
//        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
//        channels.push_back(channel);
//        data += mean_blob.height() * mean_blob.width();
//    }
//    /* Merge the separate channels into a single image. */
//    cv::Mat mean;
//    cv::merge(channels, mean);
//    /* Compute the global mean pixel value and create a mean image
//    * filled with this value. */
//    cv::Scalar channel_mean = cv::mean(mean);
//    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
//    channels.clear();
//}

