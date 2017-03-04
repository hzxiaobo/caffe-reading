#ifndef __DETECTOR_H__
#define __DETECTOR_H__
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

class Detector {
public:
    Detector(const std::string& frontalPath, const std::string& profilePath);
    std::vector<cv::Rect> detect(const cv::Mat&);
    std::vector<cv::Rect> greedyNonMaximalSuppression(std::vector<cv::Rect>, float threshold);
    ~Detector();

private:
    struct rect_comparer {
        inline bool operator()(const cv::Rect& r1, const cv::Rect& r2) {
            return (r1.y+r1.height) > (r2.y+r2.height);
        }
    };

    float unionArea(cv::Rect r1, cv::Rect r2);
    float overlapArea(cv::Rect r1, cv::Rect r2);
    float overlapRatio(cv::Rect r1, cv::Rect r2);

    std::vector<cv::CascadeClassifier> cascades_;
};

#endif