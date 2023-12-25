#include <iostream>
#include <vector>
//#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp> // ������ͷ�ļ�

#include "inference.h"


using namespace std;
using namespace cv;

// ����һ��������������ͼƬ�ϻ���Բ
void draw_circle(Mat& img, int cx, int cy, int radius, Scalar color, int thickness) {
    // ����opencv��circle����������ͼƬ��Բ�ģ��뾶����ɫ����ȵȲ���
    circle(img, Point(cx, cy), radius, color, thickness, LINE_AA);
}



int main(int argc, char** argv)
{
    bool runOnGPU = false;

 
    // 1. ���ü��ʶ��ģ��
    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference detector("model.onnx", cv::Size(320,256), "classes.txt", runOnGPU); // classes.txt ����ȱʧ

    // 2. ����һ��VideoCapture���󣬴�����ͷ
    VideoCapture cap(0); // 0��ʾĬ�ϵ�����ͷ
    if (!cap.isOpened()) // ����Ƿ�ɹ���
    {
        cout << "Error opening camera" << endl;
        return -1;
    }

    // 3. ѭ���ش�����ͷ��ȡͼ�񣬲�����ʶ��
    Mat frame;

    // ���ڶ�֡�ļ����п��ƣ�������һ֡���һ�Σ�
    int detectThis = 0;
    int detectEveryNFrame = 1;
    
    // ���ڷ�ֹ������ֱ�����£���Ҫ�ſ������ٱպϲ�������
    std::string pre_state = "open";


    while (true)
    {
        if (!cap.read(frame)) // ��ȡһ֡ͼ��
            break; // �����ȡʧ�ܣ�����ѭ��
      
        // ��֡�������ٶ�
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


        
        // ���ӻ����Ƽ��
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
        if (cv::waitKey(30) == 27) // ����ESC���˳�
            break;
    }

    // 4. �ͷ�����ͷ��Դ
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

