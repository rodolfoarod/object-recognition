// ============================
// SVM.CPP
// ============================
#include "svm.hpp"

ObjRec::ObjRec(int nWords, int nTrainImg)
{
    this->nWords = nWords;
    this->nTrainImg = nTrainImg;
    
}

cv::Mat ObjRec::getDescriptors()
{
    std::ostringstream oss;

    cv::Mat image;
    cv::Mat descriptors;
    cv::Mat allDescriptors; 
    std::vector<cv::KeyPoint> keypoints; 

    std::cout << "[Object Recognition] Features Extraction" << std::endl;

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SIFT");
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SIFT");

    int n_img = (this->nTrainImg * 0.8);
    for(int i=1; i<=n_img; i++)
    {
        std::cout << i << "/" << this->nTrainImg << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/" << i << ".png";
        std::string fileName = oss.str();

        image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        if(!image.data)                          
        {
            std::cout <<  "Could not open or find the image" << std::endl;
            continue;
        }
        
        // Detects features in an image
        detector->detect(image, keypoints);

        // Computes the descriptors for a set of keypoints detected in an image
        extractor->compute(image, keypoints, descriptors);

        // Store features descritors
        allDescriptors.push_back(descriptors);
    }

    return allDescriptors;

}

cv::Mat ObjRec::getVocabulary(const cv::Mat& descriptors)
{
    cv::Mat vocabulary;

    std::cout << "[Object Recognition] BoW - training visual vocabulary" << std::endl;
    std::cout << "[Object Recognition] BoW - Clustering " << descriptors.rows << " features" << std::endl;

    // number of words
    int clusterCount = this->nWords;
    
    // the number of times the algorithm is executed using different initial labellings
    int attempts = 1;

    // Use kmeans++ center initialization
    int flags = cv::KMEANS_PP_CENTERS;

    // kmeans -based class to train visual vocabulary using the bag of visual words approach
    cv::BOWKMeansTrainer bowTrainer(clusterCount, cv::TermCriteria(), attempts, flags);
    
    // input descriptors are clustered, returns the vocabulary
    vocabulary = bowTrainer.cluster(descriptors);

    return vocabulary;
}

void ObjRec::prepareSVMtrainData(const cv::Mat& vocabulary, cv::Mat& trainData, cv::Mat& trainLabels)
{
    std::cout << "[Object Recognition] BoW - Getting BoW histogram for each training images" << std::endl;

    // Detects keypoints in an image
    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SIFT");

    // Descriptor extractor that is used to compute descriptors for an input image and its keypoints
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SIFT");
    
    // Descriptor matcher that is used to find the nearest word of the trained vocabulary 
    // for each keypoint descriptor of the image
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

    // compute an image descriptor using the bag of visual words
    // 1 - Compute descriptors for a given image and its keypoints set.
    // 2 - Find the nearest visual words from the vocabulary for each keypoint descriptor.
    // 3 - Compute the bag-of-words image descriptor as is 
    // a normalized histogram of vocabulary words encountered in the image
    cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);

    // Sets a visual vocabulary
    // Each row of the vocabulary is a visual word (cluster center)
    bowDE.setVocabulary(vocabulary);

    std::ostringstream oss;
    cv::Mat image;
    cv::Mat bowDescriptor;
    std::vector<cv::KeyPoint> keypoints;

    // Open csv file with labels for each train image
    std::ifstream infile("img/trainLabels.csv");

    // Remove header from csv file
    std::string s;
    std::getline(infile, s);

    int n_img = (this->nTrainImg * 0.8);
    for(int i=1; i<=n_img; i++)
    {
        std::cout << i << "/" << this->nTrainImg << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/" << i << ".png";
        std::string fileName = oss.str();

        image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        if(!image.data)                          
        {
            std::cout <<  "Could not open or find the image" << std::endl;
            continue;
        }

        // Detects features in an image
        detector->detect(image, keypoints);

        // TODO: CHANGE THIS!!!!!!!
        if(keypoints.empty())
        {
            std::getline(infile, s);
            std::cout <<  "Could not find keypoints in the image" << std::endl;
            continue;
        }

        // Computes an image descriptor using the set visual vocabulary.
        bowDE.compute(image, keypoints, bowDescriptor);
        trainData.push_back(bowDescriptor);

        // Store train labels
        if(!std::getline(infile, s))
        {   
            std::cout << "Unable to get label line" << std::endl;
            continue;
        }

        std::istringstream ss(s);
        std::getline(ss, s, ',');
        std::getline(ss, s, ',');

        //std::cout << s << " = " << getLabelVal(s) << std::endl;
        trainLabels.push_back((float) getLabelVal(s));      

    }

}

int ObjRec::getLabelVal(std::string label)
{
    int labelVal = -1;

    if (label.compare("airplane") == 0)
    {
        labelVal = 0;
    }

    if (label.compare("automobile") == 0)
    {
        labelVal = 1;
    }

    if (label.compare("bird") == 0)
    {
        labelVal = 2;
    }

    if (label.compare("cat") == 0)
    {
        labelVal = 3;
    }
    
    if (label.compare("deer") == 0)
    {
        labelVal = 4;
    }

    if (label.compare("dog") == 0)
    {
        labelVal = 5;
    }

    if (label.compare("frog") == 0)
    {
        labelVal = 6;
    }

    if (label.compare("horse") == 0)
    {
        labelVal = 7;
    }

    if (label.compare("ship") == 0)
    {
        labelVal = 8;
    }

    if (label.compare("truck") == 0)
    {
        labelVal = 9;
    }

    return labelVal;

}

int ObjRec::trainSVM(const cv::Mat& trainData, const cv::Mat& trainLabels, cv::SVM& svm)
{
    std::cout << "[Object Recognition] Training SVM" << std::endl;

    // SVM Parameters
    CvSVMParams params;

    // Train SVM
    svm.train(trainData, trainLabels, cv::Mat(), cv::Mat(), params);

    // Store trained SVM
    // TODO: CHANGE THIS!!!!!
    // svm.save("train_model.svm");

    return 0;
}

int ObjRec::testSVM(const cv::Mat& vocabulary, const cv::SVM& svm)
{
    std::cout << "[Object Recognition] Testing SVM" << std::endl;

    cv::Mat testData;
    cv::Mat testLabels;
    cv::Mat testClass;

    // Detects keypoints in an image
    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SIFT");

    // Descriptor extractor that is used to compute descriptors for an input image and its keypoints
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SIFT");
    
    // Descriptor matcher that is used to find the nearest word of the trained vocabulary 
    // for each keypoint descriptor of the image
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

    // compute an image descriptor using the bag of visual words
    // 1 - Compute descriptors for a given image and its keypoints set.
    // 2 - Find the nearest visual words from the vocabulary for each keypoint descriptor.
    // 3 - Compute the bag-of-words image descriptor as is 
    // a normalized histogram of vocabulary words encountered in the image
    cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);

    // Sets a visual vocabulary
    // Each row of the vocabulary is a visual word (cluster center)
    bowDE.setVocabulary(vocabulary);

    std::ostringstream oss;
    cv::Mat image;
    cv::Mat bowDescriptor;
    std::vector<cv::KeyPoint> keypoints;

    // Open csv file with labels for each train image
    std::ifstream infile("img/trainLabels.csv");

    // Remove header from csv file
    std::string s;
    std::getline(infile, s);

    // TODO: CHANGE THIS!!!!!!!
    int n_img = (this->nTrainImg * 0.8);
    
    for(int i=1; i<=n_img; i++)
    {
        std::getline(infile, s);
    }

    for(int i=n_img+1; i<=n_img+(this->nTrainImg * 0.2); i++)
    {
        std::cout << i << "/" << this->nTrainImg << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/" << i << ".png";
        std::string fileName = oss.str();

        image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        if(!image.data)                          
        {
            std::cout <<  "Could not open or find the image" << std::endl;
            continue;
        }

        // Detects features in an image
        detector->detect(image, keypoints);

        // TODO: CHANGE THIS!!!!!!!
        if(keypoints.empty())
        {
            std::getline(infile, s);
            std::cout <<  "Could not find keypoints in the image" << std::endl;
            continue;
        }

        // Computes an image descriptor using the set visual vocabulary.
        bowDE.compute(image, keypoints, bowDescriptor);
        testData.push_back(bowDescriptor);

        // Store train labels
        if(!std::getline(infile, s))
        {
            continue;
        }

        std::istringstream ss(s);
        std::getline(ss, s, ',');
        std::getline(ss, s, ',');

        // std::cout << s << " = " << getLabelVal(s) << std::endl;
        testLabels.push_back((float) getLabelVal(s));

        // Test SVM
        float classification = svm.predict(bowDescriptor);
        testClass.push_back(classification);

    }

    // Calculate classification rate
    double rate = 1 - ((double) cv::countNonZero(testLabels - testClass) / testData.rows);
    std::cout << "[Object Recognition] Classification Rate = " << rate << std::endl;

    return 0;

}