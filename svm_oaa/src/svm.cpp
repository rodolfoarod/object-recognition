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

        // TODO: CHANGE THIS!!!!!!!
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
                trainLabels.push_back((float)1);
                counterPos++;
            }
            else
            {
                if (counterNeg < counterPos)
                {
                    trainData.push_back(bowDescriptor);
                    trainLabels.push_back((float)-1);
                    counterNeg++;
                }
            }
        }
        else
        {
            if (s.compare(classLabel) == 0)
            {
                trainData.push_back(bowDescriptor);
                trainLabels.push_back((float)1);
                counterPos++;
            }
            else
            {
                trainData.push_back(bowDescriptor);
                trainLabels.push_back((float)-1);
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
    int n_distance = 10;
    float distances[n_distance];
    float distance = 0.0;

    int matchCounter[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int labelCounter[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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

        // TODO: CHANGE THIS!!!!!!!
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
        distance = svmVec[0]->predict(bowDescriptor, true);
        distances[0] = distance;

        // automobile
        distance = svmVec[1]->predict(bowDescriptor, true);
        distances[1] = distance;

        // bird
        distance = svmVec[2]->predict(bowDescriptor, true);
        distances[2] = distance;

        // cat
        distance = svmVec[3]->predict(bowDescriptor, true);
        distances[3] = distance;
        
        // deer
        distance = svmVec[4]->predict(bowDescriptor, true);
        distances[4] = distance;

        // dog
        distance = svmVec[5]->predict(bowDescriptor, true);
        distances[5] = distance;

        // frog
        distance = svmVec[6]->predict(bowDescriptor, true);
        distances[6] = distance;

        // horse
        distance = svmVec[7]->predict(bowDescriptor, true);
        distances[7] = distance;

        // ship
        distance = svmVec[8]->predict(bowDescriptor, true);
        distances[8] = distance;

        // truck
        distance = svmVec[9]->predict(bowDescriptor, true);
        distances[9] = distance;

        // Get SVM classification
        int classification = -1;
        float dist = std::numeric_limits<float>::max();

        for(int z=0; z<n_distance; z++)
        {
            if(distances[z] < dist)
            {
                classification = z;
                dist = distances[z];
            }
        }

        labelCounter[getLabelVal(s)]++;
        if(classification == getLabelVal(s))
        {
            matchCounter[getLabelVal(s)]++;
        }

        testClass.push_back(classification);

    }

    for(int r=0; r<10; r++)
    {
        std::cout << "Class " << r << " = " << matchCounter[r] << "/" << labelCounter[r] << " = " 
<< (double)matchCounter[r] / labelCounter[r] << std::endl;
    }

    // Calculate classification rate
    double rate = 1 - ((double)cv::countNonZero(testLabels - testClass) / testData.rows);
    std::cout << "[Object Recognition] Classification Rate = " << rate << std::endl;

    return 0;
}