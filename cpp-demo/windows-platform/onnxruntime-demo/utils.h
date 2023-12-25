#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
 
struct OutputDet{
    int id;
    float confidence;
    cv::Rect box;
};
 
void DrawPred(cv::Mat& img, std::vector<OutputDet> result,
              std::vector<std::string> classNames, std::vector<cv::Scalar> color);
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
               cv::Vec4d& params,
               const cv::Size& newShape = cv::Size(640, 640),
               bool autoShape = false,
               bool scaleFill=false,
               bool scaleUp=true,
               int stride= 32,
               const cv::Scalar& color = cv::Scalar(114,114,114));