#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

#include <unistd.h>

using namespace std;
using namespace cv;

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()


void printf_red(const char *s)
{
    printf("\033[0m\033[1;31m%s\033[0m", s);
}

 

int main(int argc, char **argv)
{
    if(argc != 3){
        printf_red("Usage: [video_in] [video_out]\n");
        return 1;
    }
    std::string projectBasePath = "/home/cl/detect/yolov8/yolov8_cpp_inference/"; // Set your ultralytics base path

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //


    Inference inf(projectBasePath + "data/yolov8x_custom.onnx", cv::Size(640, 640), projectBasePath+"data/classes.txt", runOnGPU); // yolov8s

    VideoCapture cap(argv[1]);
     
    // Exit if video is not opened
    if(!cap.isOpened())
    {
        cout << "Could not read video file: {" << argv[1] << "}" << endl; 
        return 1; 
    } 

   

    Mat frame ;
    cap.read(frame);

    // This is only for preview purposes
    float scale = 0.5;

     // Video writer 
    std::string filePath = argv[2]; // 写如文件路径
	int fps=30;
	int width=frame.cols;
	int height=frame.rows;
    int codec = 0x7634706d;
    cv::VideoWriter video_writer(filePath, codec, fps ,cv::Size(frame.cols*scale, frame.rows*scale));

    while(cap.read(frame))
    {
        // // Start timer
        double timer = (double)getTickCount();
              
        
        std::vector<Detection> output = inf.runInference(frame);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

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
        }
 



        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        // cv::imshow("Inference", frame);

        // Display tracker type on frame
        putText(frame,  "YolovSeries", Point(50,150), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(50,100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        cout << "fps: " << fps << endl; 
        video_writer.write(frame);
        // cv::waitKey(10);
    }
    cap.release();
    video_writer.release();
    cout << "result video saved to : " << argv[2] << endl; 
}
