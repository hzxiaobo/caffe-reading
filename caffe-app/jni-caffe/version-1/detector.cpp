#include <iostream>
#include "detector.h"

Detector::Detector(const std::string& frontalPath, const std::string& profilePath) {
    cv::CascadeClassifier frontal_classifier;
    frontal_classifier.load(frontalPath);

    cv::CascadeClassifier profile_classifier;
    profile_classifier.load(profilePath);

    cascades_.push_back(frontal_classifier);
    cascades_.push_back(profile_classifier);
}

std::vector<cv::Rect> Detector::detect(const cv::Mat& img) {
    std::vector<cv::Rect> faces;
    cv::Mat gray_img;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray_img, CV_BGR2GRAY);
    } else {
        gray_img = img.clone();
    }
    for (int i = 0; i < cascades_.size(); ++i) {
        cv::CascadeClassifier face_cascade = cascades_[i];
        std::vector<cv::Rect> result;
        face_cascade.detectMultiScale(gray_img, result, 1.2, 5, 0|CV_HAAR_SCALE_IMAGE, cv::Size(10, 10));
        // std::cout << "size: " << result.size() << std::endl;
        for (int j = 0; j < result.size(); ++j) {
            cv::Rect rect = result[j];
            // std::cout << "p1: (" << rect.x << "," << rect.y << "), p2: (" << rect.x+rect.width << "," << rect.y + rect.height <<")" << std::endl;
        }
        faces.insert(faces.end(), result.begin(), result.end());
    }

    faces = greedyNonMaximalSuppression(faces, 0.5f);
    std::cout << "after merge" << std::endl;
    for (int i = 0; i < faces.size(); ++i) {
        cv::Rect rect = faces[i];
        std::cout << "p1: (" << rect.x << "," << rect.y << "), p2: (" << rect.x+rect.width << "," << rect.y + rect.height <<")" << std::endl;
    }
    return faces;
}


float Detector::overlapArea(cv::Rect r1, cv::Rect r2) {
    float tlxmax = (float) std::max(r1.x, r2.x);
    float tlymax = (float) std::max(r1.y, r2.y);
    float brxmin = (float) std::min(r1.x + r1.width, r2.x + r2.width);
    float brymin = (float) std::min(r1.y + r1.height, r2.y + r2.height);
    return (brymin - tlymax) * (brxmin - tlxmax);
}


float Detector::unionArea(cv::Rect r1, cv::Rect r2) {
    float a1 = (float) r1.height * r1.width;
    float a2 = (float) r2.height * r2.width;
    return a1 + a2 - overlapArea(r1, r2);
}

float Detector::overlapRatio(cv::Rect r1, cv::Rect r2) {
    return overlapArea(r1, r2) / unionArea(r1, r2);
}


std::vector<cv::Rect> Detector::greedyNonMaximalSuppression(std::vector<cv::Rect> faces, float threshold) {
    std::sort(faces.begin(), faces.end(), rect_comparer());
    std::vector<bool> removed_flag(faces.size(), false);
    std::vector<cv::Rect> merged_faces;

    for (int i = 0; i < faces.size(); ++i) {
        if (removed_flag[i]) {
            continue;
        }
        cv::Rect current_best_box = faces[i];
        merged_faces.push_back(current_best_box);

        for (int j = i+1; j < faces.size(); ++j) {
            cv::Rect rect = faces[j];
            float ratio = overlapRatio(current_best_box, rect);
            if (ratio > threshold) {
                removed_flag[j] = true;
            }
        }
    }
    return merged_faces;
}

Detector::~Detector() { }