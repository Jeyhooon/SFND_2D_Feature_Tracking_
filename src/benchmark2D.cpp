#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <fstream>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "benchmark2D.hpp"

using namespace std;

tuple<float, float> calculate_keypoint_size_statistics(vector<cv::KeyPoint> &keypoints)
{
    Eigen::VectorXf data(keypoints.size());
    for (int i = 0; i < keypoints.size(); ++i)
    {
        data(i) = keypoints[i].size;        // type of the kpt.size is float
    }
    float mean = data.mean();
    float stddev = sqrt((data.array() - mean).square().sum() / (data.size() - 1));

    return make_tuple(mean, stddev);
}


BenchData benchmark(std::string detectorType, std::string descriptorType, std::string matcherType, std::string selectorType, bool bFocusOnVehicle, bool bLimitKpts)
{
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer;

    vector<int> numDetectKpts;
    vector<int> numMatchKpts;
    vector<double> timeDetectKpts;
    vector<double> timeDetectAndDescribeKpts;
    vector<float> sizeMeanKpt;
    vector<float> sizeStdKpt;

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if (dataBuffer.size() < dataBufferSize)
        {
            dataBuffer.push_back(frame);
        }
        else
        {
            // this shifts elements in the vector one to the left (first element goes to the end)
            rotate(dataBuffer.begin(), dataBuffer.begin() + 1, dataBuffer.end());
            dataBuffer.back() = frame;      // replacing the last element with new frame
        }

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        // string detectorType = "BRISK";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT


        bool bVisKeypoints = false;
        double t1 = (double)cv::getTickCount();
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, bVisKeypoints);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, bVisKeypoints);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, bVisKeypoints);
        }
        t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
        timeDetectKpts.push_back(t1*1000);

        // only keep keypoints on the preceding vehicle
        // bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            // ToDo: remove keypoints that are outside the rectangle
            vector<cv::KeyPoint> keypointsInsideRect;
            for (auto &kpt : keypoints)
            {
                if (vehicleRect.contains(kpt.pt))
                {
                    keypointsInsideRect.push_back(kpt);
                }
            }
            keypoints = keypointsInsideRect;   // replace old keypoints
        }

        numDetectKpts.push_back(keypoints.size());
        tuple<float, float> kptSizeMeanStd = calculate_keypoint_size_statistics(keypoints);
        sizeMeanKpt.push_back(get<0>(kptSizeMeanStd));
        sizeStdKpt.push_back(get<1>(kptSizeMeanStd));
        

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        // bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        // string descriptorType = "SIFT"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        double t2 = (double)cv::getTickCount();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        t2 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();
        timeDetectAndDescribeKpts.push_back((t1 + t2)*1000);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            // string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            // string selectorType = "SEL_KNN";         // SEL_NN, SEL_KNN
            
            // DES_BINARY, DES_HOG  (this is important when using Brute-Force matching)
            if (descriptorType.compare("SIFT") == 0 || descriptorType.compare("AKAZE") == 0)
            {
                // SIFT and AKAZE output descriptors as real values 
                //  --> hence Norm_L2 should be used to calculate distance between descriptors for matching.
                string descriptorType = "DES_HOG";    
            } 
            else
            {
                // output binary descriptors (string of 0s and 1s) --> hence Norm_Hamming should be used.
                string descriptorType = "DES_BINARY";
            }

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            numMatchKpts.push_back(matches.size());

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
        }
        

    } // eof loop over all images

    BenchData benchData;
    benchData.numDetectKpts = accumulate(numDetectKpts.begin(), numDetectKpts.end(), 0);
    benchData.numMatchKpts = accumulate(numMatchKpts.begin(), numMatchKpts.end(), 0);
    benchData.sizeMeanKpts = accumulate(sizeMeanKpt.begin(), sizeMeanKpt.end(), 0.0f) / sizeMeanKpt.size();
    benchData.sizeStdKpts = accumulate(sizeStdKpt.begin(), sizeStdKpt.end(), 0.0f) / sizeStdKpt.size();
    benchData.timeDetectKpts = accumulate(timeDetectKpts.begin(), timeDetectKpts.end(), 0.0);
    benchData.timeDetectAndMatchKpts = accumulate(timeDetectAndDescribeKpts.begin(), timeDetectAndDescribeKpts.end(), 0.0);

    return benchData;
}

int main()
{
    vector<string> detectorList = {"SHITOMASI", "HARRIS", "SIFT", "FAST", "BRISK", "ORB", "AKAZE"};
    vector<string> descriptorList = {"BRISK", "BRIEF", "ORB", "FREAK", "SIFT", "AKAZE"};

    // Open the CSV file
    std::ofstream file("../benchmark_data.csv");
    file << "detectorType,descriptorType,numDetectKpts,numMatchKpts,timeDetectKpts,timeDetectAndMatchKpts,sizeMeanKpts,sizeStdKpts\n";


    for (string detectorType : detectorList)
    {
        for (string descriptorType : descriptorList)
        {
            try
            {
            cout << "Benchmarking: " << detectorType << " (detector), " << descriptorType << " (descriptor)" << endl;
            BenchData benchmarkData = benchmark(detectorType, descriptorType);

            // Write data immediately after benchmarking
            file << detectorType << "," << descriptorType << ",";
            file << benchmarkData.numDetectKpts << "," << benchmarkData.numMatchKpts << ",";
            file << benchmarkData.timeDetectKpts << "," << benchmarkData.timeDetectAndMatchKpts << ",";
            file << benchmarkData.sizeMeanKpts << "," << benchmarkData.sizeStdKpts << "\n";
            }
            catch (cv::Exception& e)
            {
                std::cerr << "Caught OpenCV exception: " << e.what() << std::endl;
                file << detectorType << "," << descriptorType << ",";
                file << "-,-,-,-,-,-" << "\n";
            }

            
        }
    }

    return 0;
}