// ============================
// MAIN.CPP
// ============================

#include "svm.hpp"

// Main Function
int main(int argc, const char* argv[]) {

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    cv::initModule_nonfree();

    ObjRec obj_rec(10,1000);
    cv::Mat descriptors = obj_rec.getDescriptors();
    cv::Mat vocabulary = obj_rec.getVocabulary(descriptors);

    cv::Mat trainData;
    cv::Mat trainLabels;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels);

    //std::cout << trainData << std::endl;
    //std::cout << trainLabels << std::endl;

    // Support Vector Machines
    cv::SVM svm;

    // Train SVM
    obj_rec.trainSVM(trainData, trainLabels, svm);

    // Test SVM
    obj_rec.testSVM(vocabulary, svm);

    return 0;
}