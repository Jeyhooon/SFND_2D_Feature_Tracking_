#ifndef benchmark2D_hpp     // safe-guard
#define benchmark2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>

struct BenchData {
    int numDetectKpts;
    int numMatchKpts;
    double timeDetectKpts;
    double timeDetectAndMatchKpts;
    float sizeMeanKpts;
    float sizeStdKpts;
};

std::tuple<float, float> calculate_keypoint_size_statistics(std::vector<cv::KeyPoint> &keypoints);
BenchData benchmark(std::string detectorType, std::string descriptorType, std::string matcherType = "MAT_BF", std::string selectorType = "SEL_KNN", bool bFocusOnVehicle = true, bool bLimitKpts = false);

#endif  /* benchmark2D_hpp */