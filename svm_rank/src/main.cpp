// ============================
// Multiclass Ranking SVM
// ============================
// ============================
// MAIN.CPP
// ============================

#include "svm.hpp"

// Main Function
int main(int argc, const char* argv[]) {

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    if (argc != 3) {
		std::cout << "Usage: ./obj_rec_svm nWords nTrainImg" << std::endl;
		exit(-1);
	}

    cv::initModule_nonfree();

    int nWords = atoi(std::string(argv[1]).c_str());
    int nTrainImg = atoi(std::string(argv[2]).c_str());

    ObjRec obj_rec(nWords,nTrainImg);
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