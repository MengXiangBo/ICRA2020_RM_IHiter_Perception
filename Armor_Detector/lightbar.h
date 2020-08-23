//
// Created by mrm on 19-11-13.
//

#ifndef ARMOR_DETECTOR_LIGHTBAR_H
#define ARMOR_DETECTOR_LIGHTBAR_H

#include <string.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace hitcrt{

enum ID{
    red = 0,
    blue = 1
};

enum STATE{
    level = 0,
    verticle = 1
};

class Lightbar{
public:
    ID id;
    STATE state;
    double lightbar_z = 128.457; // mm
    double lightbar_width = 135; // mm
    double lightbar_height = 125; // mm
    // square_width
    // ________
    // |    \ | leap_lightbar_dist
    // | robo\| square_height
    // |      |
    // |______|
    double leap_lightbar_dist = 235.704; // mm
    double square_height = 536.696; // mm
    double square_width = 375.696; // mm
    double contour_area; // pixel
    double cigma_ratio = 550; // cigma_x = area/cigma_ratio总之保证area = 110的时候,cigmax = 0.2,这个值是对于1280x720的
    double cigma_y; // cigma_y = 5*cigma_x
    double cigma_x;
    double length;
    double width;
    double ratio; // length/width
    double coloratio; // white/total in gray image
    double k,b,L; // 斜率及截距
    double theta;
    bool is_overlap = false; // 是否有重叠
    cv::Point2d up,down; // 上交点、下交点
    cv::Point2d focus;
    cv::Point2f upcenter = cv::Point2f(0.0,0.0);
    cv::Point2f downcenter = cv::Point2f(0.0,0.0);
    cv::Point3f location; // lightbar location in camera coordinate,norm vector
    std::vector<cv::Point> contour;
    cv::RotatedRect boundingbox;

    Lightbar(const ID i, const std::vector<cv::Point > &c,cv::Mat &image):
            id(i),contour(c){
        // std::vector<std::vector<cv::Point >> contours;
        // contours.push_back(contour);
        /*
        cigma_x = cv::contourArea(contour)/cigma_ratio;
        if (cigma_x > 0.5)
            cigma_x = 0.5;
        cigma_y = 4*cigma_x;
         */
        cigma_x = 3;
        cigma_y = 18;
        boundingbox = cv::minAreaRect(contour);
        // boundingbox = cv::fitEllipse(contour);
        calc_parameter(image);
    }
    ~Lightbar() = default;

private:

	// 计算灯条在图像中的位置
    void calc_parameter(const cv::Mat &image);
};
}

#endif //ARMOR_DETECTOR_LIGHTBAR_H
