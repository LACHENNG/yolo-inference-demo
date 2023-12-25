 
#include <iostream>
#include <opencv2/opencv.hpp>
#include "inference.h"

 
using namespace std;
using namespace cv;
using namespace cv::dnn;
 
int main(){
 
    string detect_model_path = "model.onnx";
    Yolov8Onnx yolov8;
    if (yolov8.ReadModel(detect_model_path))
        cout << "read Net ok!\n";
    else {
        return -1;
    }
 
    //生成随机颜色；每个类别都有自己的颜色
    vector<Scalar> color;
    srand((time(0)));
    for (int i=0; i<80; i++){
        int b = rand() %  256; //随机数为0～255
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b,g,r));
    }
 
 
    VideoCapture capture(0);
 
    Mat frame;

    double timeuse;
    while (1) {
        capture>>frame;
        vector<OutputDet> reuslt;
 
        bool  find = yolov8.OnnxDetect(frame, reuslt);
  
 
        if(find)
        {
            DrawPred(frame, reuslt, yolov8._className, color);
        }
        else {
            cout << "Don't find !\n";
        }
 
        // timeuse ? 
        // string label = "duration:" + to_string(timeuse*1000); // ms
        string label = "duration:" + to_string(0 * 1000); // ms
        putText(frame, label, Point(30,30), FONT_HERSHEY_SIMPLEX,
                0.5, Scalar(0,0,255), 2, 8);
 
        imshow("result", frame);
        if (waitKey(1)=='q')  break;
 
    }
 
 
    return 0;
}