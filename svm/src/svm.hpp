// ============================
// SVM.HPP
// ============================
#ifndef svm_hpp
#define svm_hpp

// #include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

class ObjRec
{
public:
    cv::Mat dictionary;
    int nWords;
    int nTrainImg;

    ObjRec(int nWords, int nTrainImg);

    // Get image features descriptors for each training images
    cv::Mat getDescriptors();

    // kmeans -based class to train visual vocabulary using the bag of visual words approach
    // Clusters train descriptors
    // The vocabulary consists of cluster centers.
    cv::Mat getVocabulary(const cv::Mat& descriptors);

    // Get BoW histogram for each training images
    void prepareSVMtrainData(const cv::Mat& vocabulary, cv::Mat& trainData, cv::Mat& trainLabels);

    // Get int value for image label
    int getLabelVal(std::string label);

};

#endif /* svm_hpp */