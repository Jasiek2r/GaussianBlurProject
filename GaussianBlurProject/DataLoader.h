#pragma once
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

class DataLoader {
private:
	cv::Mat image;
public:
	void Load(string path);
	bool HasImage();
	cv::Mat GetImage();
};