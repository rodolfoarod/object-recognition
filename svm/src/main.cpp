// ============================
// MAIN.CPP
// ============================

#include <iostream>

#include <opencv2/opencv.hpp>

#include "svm.hpp"

// Main Function
int main(int argc, const char* argv[]) {

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    loadImgBatch();
    
    return 0;
}