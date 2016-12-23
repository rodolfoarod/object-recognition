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

    // Support Vector Machines - One vs. All

    cv::Mat trainData;
    cv::Mat trainLabels;
    std::vector<cv::SVM*> svmVec;
    
    // airplane
    cv::SVM airplaneSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "airplane");
    obj_rec.trainSVM(trainData, trainLabels, airplaneSVM);
    svmVec.push_back(&airplaneSVM);

    // automobile
    cv::SVM automobileSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "automobile");
    obj_rec.trainSVM(trainData, trainLabels, automobileSVM);
    svmVec.push_back(&automobileSVM);

    // bird
    cv::SVM birdSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "bird");
    obj_rec.trainSVM(trainData, trainLabels, birdSVM);
    svmVec.push_back(&birdSVM);

    // cat
    cv::SVM catSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "cat");
    obj_rec.trainSVM(trainData, trainLabels, catSVM);
    svmVec.push_back(&catSVM);

    // deer
    cv::SVM deerSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "deer");
    obj_rec.trainSVM(trainData, trainLabels, deerSVM);
    svmVec.push_back(&deerSVM);

    // dog
    cv::SVM dogSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "dog");
    obj_rec.trainSVM(trainData, trainLabels, dogSVM);
    svmVec.push_back(&dogSVM);

    // frog
    cv::SVM frogSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "frog");
    obj_rec.trainSVM(trainData, trainLabels, frogSVM);
    svmVec.push_back(&frogSVM);

    // horse
    cv::SVM horseSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "horse");
    obj_rec.trainSVM(trainData, trainLabels, horseSVM);
    svmVec.push_back(&horseSVM);

    // ship
    cv::SVM shipSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "ship");
    obj_rec.trainSVM(trainData, trainLabels, shipSVM);
    svmVec.push_back(&shipSVM);

    // truck
    cv::SVM truckSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "truck");
    obj_rec.trainSVM(trainData, trainLabels, truckSVM);
    svmVec.push_back(&truckSVM);

    // Test SVM
    obj_rec.testSVM(vocabulary, svmVec);

    return 0;
}