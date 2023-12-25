#pragma once
#include "utils.h"
using namespace cv;
using namespace std;
 
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
               cv::Vec4d& params,
               const cv::Size& newShape,
               bool autoShape,
               bool scaleFill,
               bool scaleUp,
               int stride,
               const cv::Scalar& color)
{
    if (false) {
        int maxLen = MAX(image.rows, image.cols);
        outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[3] = 0;
        params[2] = 0;
    }
 
    // 取较小的缩放比例
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);
 
    // 依据前面的缩放比例后，原图的尺寸
    float ratio[2]{r,r};
    int new_un_pad[2] = { (int)std::round((float)shape.width  * r), (int)std::round((float)shape.height * r)};
 
    // 计算距离目标尺寸的padding像素数
    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);
    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }
 
    dw /= 2.0f;
    dh /= 2.0f;
 
    // 等比例缩放
    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else{
        outImage = image.clone();
    }
 
    // 图像四周padding填充，至此原图与目标尺寸一致
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0]; // width的缩放比例
    params[1] = ratio[1]; // height的缩放比例
    params[2] = left; // 水平方向两边的padding像素数
    params[3] = top; //垂直方向两边的padding像素数
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}
 
void DrawPred(cv::Mat& img, std::vector<OutputDet> result,
              std::vector<std::string> classNames,
              std::vector<cv::Scalar> color)
{
    for (size_t i=0; i<result.size(); i++){
        int  left,top;
        left = result[i].box.x;
        top = result[i].box.y;
 
        // 框出目标
        rectangle(img, result[i].box,color[result[i].id], 2, 8);
 
        // 在目标框左上角标识目标类别以及概率
        string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
    }
}
 