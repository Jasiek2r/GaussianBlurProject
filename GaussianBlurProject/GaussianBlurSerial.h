#pragma once
#include <opencv2/opencv.hpp>

class GaussianBlurSerial
{
public:
    static cv::Mat Process(const cv::Mat& inputImage, int kernelSize, double sigma = 0.0);
};