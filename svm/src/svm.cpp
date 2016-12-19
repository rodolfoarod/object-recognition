// ============================
// SVM.CPP
// ============================
#include "svm.hpp"

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

// Load image batch file
int loadImgBatch()
{
    std::streampos size;
    char* memblock;

    // Read binary file
    std::ifstream file("img/data_batch_1.bin", std::ios::in|std::ios::binary|std::ios::ate);

    if(file.is_open())
    {   
        size = file.tellg();
        std::cout << "File size = " << size << " bytes." << std::endl;

        // Allocation of a memory block
        memblock = new char[size];

        // Position at the beginning of the file and read entire file
        file.seekg(0, std::ios::beg);
        file.read(memblock, size);
        file.close();

        std::cout << "Image Label = " << (int)memblock[0] << std::endl;

        // Build Image
        char* temp;
        temp = new char[3073];
        for(size_t i=0; i<3073; i++)
        {
            temp[i] = memblock[i+1];
        }

        cv::Mat imgData = cv::Mat::zeros(32,32, CV_8UC3);
        
        
        
        cv::imshow("My Window", imgData);
    
        cv::waitKey(0);

        delete[] memblock;

    }
    else std::cout << "Unable to open file" << std::endl;

    return 0;
}