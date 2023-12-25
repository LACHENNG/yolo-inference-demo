#include <iostream>
#include <vector>
//#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp> // 添加这个头文件

#include "inference.h"


using namespace std;
using namespace cv;

// 定义一个函数，用于在图片上绘制圆
void draw_circle(Mat& img, int cx, int cy, int radius, Scalar color, int thickness) {
    // 调用opencv的circle函数，传入图片，圆心，半径，颜色，厚度等参数
    circle(img, Point(cx, cy), radius, color, thickness, LINE_AA);
}



int main(int argc, char** argv)
{
    bool runOnGPU = false;

 
    // 1. 设置检测识别模型
    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference detector("model.onnx", cv::Size(320,256), "classes.txt", runOnGPU); // classes.txt 可以缺失

    // 2. 创建一个VideoCapture对象，打开摄像头
    VideoCapture cap(0); // 0表示默认的摄像头
    if (!cap.isOpened()) // 检查是否成功打开
    {
        cout << "Error opening camera" << endl;
        return -1;
    }

    // 3. 循环地从摄像头读取图像，并进行识别
    Mat frame;

    // 用于对帧的检测进行控制（本例隔一帧检测一次）
    int detectThis = 0;
    int detectEveryNFrame = 1;
    
    // 用于防止连续的直接落下，需要张开手掌再闭合才能落下
    std::string pre_state = "open";


    while (true)
    {
        if (!cap.read(frame)) // 读取一帧图像
            break; // 如果读取失败，跳出循环
      
        // 跳帧检测提高速度
        detectThis = (detectThis + 1) % detectEveryNFrame;
       /* if (detectThis) {
            cv::imshow("Inference", frame);
            continue;
        }*/

        // Inference starts here...
        std::vector<Detection> output = detector.runInference(frame);
        
        int detections = output.size();
        //if (detections == 0) continue;
        // std::cout << "Number of detections:" << detections << std::endl;


        
        // 可视化手势检测
        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

            // Detection Center
            int cx = box.x + box.width / 2;
            int cy = box.y + box.height / 2;
            draw_circle(frame, cx, cy, 10, Scalar(0, 0, 255), 2);

        }
        cv::imshow("Inference", frame);
        if (cv::waitKey(30) == 27) // 按下ESC键退出
            break;
    }

    // 4. 释放摄像头资源
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

