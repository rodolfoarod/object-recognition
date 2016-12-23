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
    for (int i = 1; i <= n_img; i++)
    {
        std::cout << i << "/" << this->nTrainImg << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/" << i << ".png";
        std::string fileName = oss.str();

        image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        if (!image.data)
        {
            std::cout << "Could not open or find the image" << std::endl;
            continue;
        }

        // Detects features in an image
        detector->detect(image, keypoints);

        if(keypoints.empty())
        {
            continue;
        }

        // Computes the descriptors for a set of keypoints detected in an image
        extractor->compute(image, keypoints, descriptors);

        // Store features descritors
        allDescriptors.push_back(descriptors);
    }

    return allDescriptors;
}

cv::Mat ObjRec::getVocabulary(const cv::Mat &descriptors)
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

void ObjRec::prepareSVMtrainData(const cv::Mat &vocabulary, cv::Mat &trainData, cv::Mat &trainLabels, std::string classLabel, bool balanced)
{
    std::cout << "[Object Recognition] Train data for SVM: " << classLabel << std::endl;

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

    int counterPos = 0;
    int counterNeg = 0;

    int n_img = (this->nTrainImg * 0.8);
    for (int i = 1; i <= n_img; i++)
    {
        // TODO: CHANGE THIS!!!!
        //std::cout << i << "/" << this->nTrainImg << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/" << i << ".png";
        std::string fileName = oss.str();

        image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        if (!image.data)
        {
            std::cout << "Could not open or find the image" << std::endl;
            continue;
        }

        // Detects features in an image
        detector->detect(image, keypoints);

        if (keypoints.empty())
        {
            std::getline(infile, s);
            //std::cout << "Could not find keypoints in the image" << std::endl;
            continue;
        }

        // Computes an image descriptor using the set visual vocabulary.
        bowDE.compute(image, keypoints, bowDescriptor);

        // Store train labels
        if (!std::getline(infile, s))
        {
            std::cout << "Unable to get label line" << std::endl;
            continue;
        }

        std::istringstream ss(s);
        std::getline(ss, s, ',');
        std::getline(ss, s, ',');

        if (balanced)
        {
            if (s.compare(classLabel) == 0)
            {
                trainData.push_back(bowDescriptor);
                trainLabels.push_back(1);
                counterPos++;
            }
            else
            {
                if (counterNeg < counterPos)
                {
                    trainData.push_back(bowDescriptor);
                    trainLabels.push_back(-1);
                    counterNeg++;
                }
            }
        }
        else
        {
            if (s.compare(classLabel) == 0)
            {
                trainData.push_back(bowDescriptor);
                trainLabels.push_back(1);
                counterPos++;
            }
            else
            {
                trainData.push_back(bowDescriptor);
                trainLabels.push_back(-1);
                counterNeg++;
            }
        }
    }

    std::cout << "Pos Samples = " << counterPos << std::endl;
    std::cout << "Neg Samples = " << counterNeg << std::endl;

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

int ObjRec::trainSVM(const cv::Mat &trainData, const cv::Mat &trainLabels, cv::SVM &svm)
{
    std::cout << "[Object Recognition] Training SVM" << std::endl;

    // SVM Parameters
    CvSVMParams params;

    // Train SVM
    svm.train_auto(trainData, trainLabels, cv::Mat(), cv::Mat(), params);

    return 0;
}

int ObjRec::testSVM(const cv::Mat& vocabulary, const std::vector<cv::SVM*> svmVec)
{
    std::cout << "[Object Recognition] Testing SVM" << std::endl;

    cv::Mat testData;
    cv::Mat testLabels;
    cv::Mat testClass;
    int n_class = 10;
    double distances[n_class];

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
    for (int i = 1; i <= n_img; i++)
    {
        std::getline(infile, s);
    }

    for (int i = n_img + 1; i <= n_img + (this->nTrainImg * 0.2); i++)
    {
        std::cout << i << "/" << this->nTrainImg << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/" << i << ".png";
        std::string fileName = oss.str();

        image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        if (!image.data)
        {
            std::cout << "Could not open or find the image" << std::endl;
            continue;
        }

        // Detects features in an image
        detector->detect(image, keypoints);

        if (keypoints.empty())
        {
            std::getline(infile, s);
            //std::cout << "Could not find keypoints in the image" << std::endl;
            continue;
        }

        // Computes an image descriptor using the set visual vocabulary.
        bowDE.compute(image, keypoints, bowDescriptor);
        testData.push_back(bowDescriptor);

        // Store train labels
        if (!std::getline(infile, s))
        {
            continue;
        }

        std::istringstream ss(s);
        std::getline(ss, s, ',');
        std::getline(ss, s, ',');

        // std::cout << s << " = " << getLabelVal(s) << std::endl;
        testLabels.push_back(getLabelVal(s));

        // airplane
        distances[0] = svmVec[0]->predict(bowDescriptor, true);
        
        // automobile
        distances[1] = svmVec[1]->predict(bowDescriptor, true);
        
        // bird
        distances[2] = svmVec[2]->predict(bowDescriptor, true);
        
        // cat
        distances[3] = svmVec[3]->predict(bowDescriptor, true);
        
        // deer
        distances[4] = svmVec[4]->predict(bowDescriptor, true);
        
        // dog
        distances[5] = svmVec[5]->predict(bowDescriptor, true);
        
        // frog
        distances[6] = svmVec[6]->predict(bowDescriptor, true);
        
        // horse
        distances[7] = svmVec[7]->predict(bowDescriptor, true);
        
        // ship
        distances[8] = svmVec[8]->predict(bowDescriptor, true);
        
        // truck
        distances[9] = svmVec[9]->predict(bowDescriptor, true);
        
        // Get SVM classification
        int classification = -1;
        double dist = std::numeric_limits<double>::max();

        for(int z=0; z<n_class; z++)
        {
            if(distances[z] < dist)
            {
                classification = z;
                dist = distances[z];
            }
        }

        testClass.push_back(classification);

        // Calculate classification rate
        double rateIt = 1 - ((double)cv::countNonZero(testLabels - testClass) / testData.rows);
        std::cout << "Iteration Rate = " << rateIt << std::endl;

    }

    std::cout << "[Object Recognition] Final Results " << std::endl;

    // Calculate classification rate
    double rate = 1 - ((double)cv::countNonZero(testLabels - testClass) / testData.rows);
    std::cout << "Classification Rate = " << rate << std::endl;

    return 0;
}