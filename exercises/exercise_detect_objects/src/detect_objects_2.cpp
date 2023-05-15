#include <iostream>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "dataStructures.h"

using namespace std;

void detectObjects2()
{
    // load image from file
    cv::Mat img = cv::imread("../images/0000000000.png");

    // load class names from file
    string yoloBasePath = "../dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights"; 

    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // generate 4D blob from input image
    cv::Mat blob;
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(416, 416);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

    // Get names of output layers
    vector<cv::String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers(); // get indices of output layers, i.e. layers with unconnected outputs
    vector<cv::String> layersNames = net.getLayerNames(); // get names of all layers in the network
    
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
    {
        // '-1' is necessary because layer indices in the outLayers vector are 1-based, while C++ vectors are 0-based.
        names[i] = layersNames[outLayers[i] - 1];
    }

    // invoke forward propagation through network
    vector<cv::Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names);      // computing the output of the layers specified by 'names'. 
    // output is represented as a vector with 85 elements:
    // first 4 elements represent the center coordinates (x, y), width, and height of a bounding box (relative to image size, typically normalized -> [0, 1])
    // 5th element is the confidence score (how certain the model is that it has found an object in the bounding box)
    // remaining 80 elements are class probabilities, one for each of the 80 classes in the COCO dataset.
    // netOutput is a vector with 3 elements each one is a matrix:
    // netOutput[0]: Mat with size = (507, 85);  where 507  = 13*13*3 (for each grid cell we have 3 bounding box prediction)
    // netOutput[1]: Mat with size = (2028, 85); where 2028 = 26*26*3
    // netOutput[2]: Mat with size = (8112, 85); where 8112 = 52*52*3

    // Scan through all bounding boxes and keep only the ones with high confidence
    float confThreshold = 0.20;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        float* data = (float*)netOutput[i].data;
        // The pointer data now points to the same memory location as the netOutput[i].data pointer, 
        // but it is declared as a float*:
        // By declaring data as a float*, you can access the data elements of the matrix netOutput[i] 
        // using pointer arithmetic and treat them as float values. 
        // For example, you can use data[0] to access the first element, data[1] for the second element, and so on.

        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
            // data += netOutput[i].cols: data is being incremented by netOutput[i].cols. 
            // This statement advances the data pointer to the next row of the matrix netOutput[i]. 
            // Since data is initially pointing to the beginning of the current row, 
            //   adding netOutput[i].cols to data moves it to the start of the next row. 
            // This allows the loop to iterate over each row of the matrix and process the data accordingly.
            
            // Creates a matrix header for the specified column span; scores.size = (1, 80)
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            cv::Point classId;
            double confidence;
            
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width/2; // left
                box.y = cy - box.height/2; // top
                
                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
            }
        }
    }

    // perform non-maxima suppression
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;
    for (auto it = indices.begin(); it != indices.end(); ++it)
    {
        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        
        bBoxes.push_back(bBox);
    }
    
    
    // show results
    cv::Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it)
    {
        // Draw rectangle displaying the bounding box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);

        string label = cv::format("%.2f", (*it).confidence);
        label = classes[((*it).classID)] + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 1);
    }

    string windowName = "Object classification";
    cv::namedWindow( windowName, 1 );
    cv::imshow( windowName, visImg );
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    detectObjects2();
}