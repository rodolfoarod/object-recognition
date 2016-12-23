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

    // Support Vector Machines

    cv::Mat trainData;
    cv::Mat trainLabels;
    
    // airplane
    cv::SVM airplaneSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "airplane");
    obj_rec.trainSVM(trainData, trainLabels, airplaneSVM);
    
    // Test SVM
    obj_rec.testSVM(vocabulary, airplaneSVM, "airplane");

    return 0;
}