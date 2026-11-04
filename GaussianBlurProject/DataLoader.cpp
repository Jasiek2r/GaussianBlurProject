#include "DataLoader.h"
#include <iostream>
#include <opencv2/opencv.hpp>

void DataLoader::Load(std::string path) {
    cv::Mat loadedImage = cv::imread(path, cv::IMREAD_COLOR);

    if (loadedImage.empty()) {
        std::cerr << "Błąd DataLoader: Nie można załadować pliku z ścieżki: " << path << std::endl;
        std::cerr << "Upewnij się, że plik istnieje i ścieżka/nazwa jest poprawna." << std::endl;
        return;
    }

    this->image = loadedImage;

    std::cout << "Sukces! Załadowano obraz o wymiarach: "
        << this->image.cols << "x" << this->image.rows << " px." << std::endl;
}

bool DataLoader::HasImage() {
    return !image.empty();
}

cv::Mat DataLoader::GetImage() {
    return image;
}

