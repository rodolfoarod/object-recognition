// ============================
// MAIN.CPP
// ============================

#include "svm.hpp"

// Main Function
int main(int argc, const char* argv[]) {

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    if (argc > 4) {
		std::cout << "Usage: ./obj_rec_svm nWords nTrainImg [balanced]" << std::endl;
		exit(-1);
	}

    cv::initModule_nonfree();

    int nWords = atoi(std::string(argv[1]).c_str());
    int nTrainImg = atoi(std::string(argv[2]).c_str());
    int balanced = false;

    if((argc == 4) && (std::string(argv[3]).compare("balanced") == 0))
    {
        balanced = true;
    }

    std::cout << "Num Words = " << nWords << std::endl;
    std::cout << "Num Img   = " << nTrainImg << std::endl;
    std::cout << "Balanced  = " << balanced << std::endl;

    ObjRec obj_rec(nWords,nTrainImg);
    cv::Mat descriptors = obj_rec.getDescriptors();
    cv::Mat vocabulary = obj_rec.getVocabulary(descriptors);

    // Support Vector Machines - One vs. All

    cv::Mat trainData;
    cv::Mat trainLabels;
    std::vector<cv::SVM*> svmVec;
    
    // airplane
    cv::SVM airplaneSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "airplane", balanced);
    obj_rec.trainSVM(trainData, trainLabels, airplaneSVM);
    svmVec.push_back(&airplaneSVM);
    trainData.release();
    trainLabels.release();

    // automobile
    cv::SVM automobileSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "automobile", balanced);
    obj_rec.trainSVM(trainData, trainLabels, automobileSVM);
    svmVec.push_back(&automobileSVM);
    trainData.release();
    trainLabels.release();

    // bird
    cv::SVM birdSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "bird", balanced);
    obj_rec.trainSVM(trainData, trainLabels, birdSVM);
    svmVec.push_back(&birdSVM);
    trainData.release();
    trainLabels.release();

    // cat
    cv::SVM catSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "cat", balanced);
    obj_rec.trainSVM(trainData, trainLabels, catSVM);
    svmVec.push_back(&catSVM);
    trainData.release();
    trainLabels.release();

    // deer
    cv::SVM deerSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "deer", balanced);
    obj_rec.trainSVM(trainData, trainLabels, deerSVM);
    svmVec.push_back(&deerSVM);
    trainData.release();
    trainLabels.release();

    // dog
    cv::SVM dogSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "dog", balanced);
    obj_rec.trainSVM(trainData, trainLabels, dogSVM);
    svmVec.push_back(&dogSVM);
    trainData.release();
    trainLabels.release();

    // frog
    cv::SVM frogSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "frog", balanced);
    obj_rec.trainSVM(trainData, trainLabels, frogSVM);
    svmVec.push_back(&frogSVM);
    trainData.release();
    trainLabels.release();

    // horse
    cv::SVM horseSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "horse", balanced);
    obj_rec.trainSVM(trainData, trainLabels, horseSVM);
    svmVec.push_back(&horseSVM);
    trainData.release();
    trainLabels.release();

    // ship
    cv::SVM shipSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "ship", balanced);
    obj_rec.trainSVM(trainData, trainLabels, shipSVM);
    svmVec.push_back(&shipSVM);
    trainData.release();
    trainLabels.release();

    // truck
    cv::SVM truckSVM;
    obj_rec.prepareSVMtrainData(vocabulary, trainData, trainLabels, "truck", balanced);
    obj_rec.trainSVM(trainData, trainLabels, truckSVM);
    svmVec.push_back(&truckSVM);
    trainData.release();
    trainLabels.release();

    // Test SVM
    obj_rec.testSVM(vocabulary, svmVec);

    return 0;
}