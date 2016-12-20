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

    int n_img = this->nTrainImg;
    for(int i=1; i<=n_img; i++)
    {
        std::cout << i << "/" << n_img << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/train/" << i << ".png";
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

    int n_img = this->nTrainImg;
    for(int i=1; i<=n_img; i++)
    {
        std::cout << i << "/" << n_img << std::endl;

        // Clear string stream
        oss.str(std::string());

        // Build filename
        oss << "img/train/" << i << ".png";
        std::string fileName = oss.str();

        image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        if(!image.data)                          
        {
            std::cout <<  "Could not open or find the image" << std::endl;
            continue;
        }

        // Detects features in an image
        detector->detect(image, keypoints);

        // Computes an image descriptor using the set visual vocabulary.
        bowDE.compute(image, keypoints, bowDescriptor);
        trainData.push_back(bowDescriptor);

    }


}