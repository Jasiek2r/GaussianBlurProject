#include "GaussianBlur.h"

#include "GaussianBlur.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>

cv::Mat GaussianBlur::Process(const cv::Mat& inputImage, int kernelSize, double sigma)
{
    if (inputImage.empty() || inputImage.type() != CV_8UC3) {
        std::cerr << "Błąd: Obraz musi być niepusty i w formacie 8-bitowym RGB/BGR (CV_8UC3)." << std::endl;
        return cv::Mat();
    }
    if (kernelSize % 2 == 0) {
        // Since Gaussian filter requires
        // a window of uneven size
        // the program will increase it automatically by 1
        // if an even kernel size is given
        kernelSize++;
    }

    if (sigma <= 0.0) {
        // if the user enters a non-positive sigma or does not enter it
        // then we use a formula "borrowed" from official OpenCV algorithms
        sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
    }

    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int radius = kernelSize / 2;
    cv::Mat outputImage = inputImage.clone();

    std::vector<double> kernel(kernelSize);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double g = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel[i + radius] = g;
        sum += g;
    }
    for (int i = 0; i < kernelSize; ++i) {
        kernel[i] /= sum; // normalization
    }

    cv::Mat tempImage = inputImage.clone();


#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < rows; ++y) {
        const uchar* srcRow = inputImage.ptr<uchar>(y); // direct RAM access
        uchar* destRow = tempImage.ptr<uchar>(y); // here as well

        for (int x = 0; x < cols; ++x) {
            double bSum = 0.0, gSum = 0.0, rSum = 0.0;

            for (int k = -radius; k <= radius; ++k) {
                int kx = std::min(std::max(x + k, 0), cols - 1); // this ensures that we dont touch area outside of image
                double weight = kernel[k + radius]; // gets kernel's weight for that pixel

                bSum += srcRow[kx * 3 + 0] * weight;
                gSum += srcRow[kx * 3 + 1] * weight;
                rSum += srcRow[kx * 3 + 2] * weight;
            }

            destRow[x * 3 + 0] = static_cast<uchar>(std::min(std::max(bSum, 0.0), 255.0));
            destRow[x * 3 + 1] = static_cast<uchar>(std::min(std::max(gSum, 0.0), 255.0));
            destRow[x * 3 + 2] = static_cast<uchar>(std::min(std::max(rSum, 0.0), 255.0));
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < rows; ++y) {
        uchar* destRow = outputImage.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
            double bSum = 0.0, gSum = 0.0, rSum = 0.0;

            for (int k = -radius; k <= radius; ++k) {
                int ky = std::min(std::max(y + k, 0), rows - 1);
                const uchar* tempRow = tempImage.ptr<uchar>(ky);
                double weight = kernel[k + radius];

                bSum += tempRow[x * 3 + 0] * weight;
                gSum += tempRow[x * 3 + 1] * weight;
                rSum += tempRow[x * 3 + 2] * weight;
            }

            destRow[x * 3 + 0] = static_cast<uchar>(std::min(std::max(bSum, 0.0), 255.0));
            destRow[x * 3 + 1] = static_cast<uchar>(std::min(std::max(gSum, 0.0), 255.0));
            destRow[x * 3 + 2] = static_cast<uchar>(std::min(std::max(rSum, 0.0), 255.0));
        }
    }

    return outputImage;
}
