//
// Created by mrm on 19-11-13.
//

#ifndef ARMOR_DETECTOR_CAMERA_H
#define ARMOR_DETECTOR_CAMERA_H

#include <iostream>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace hitcrt{

enum CAMERAID{
    USBFHD01M = 0, // 1280x720
    DAHUA1 = 1, // DAHUA in 1280x1024
    DAHUA2 = 2, // DAHUA in 1280x720
    OTHER = 3 // read from file
};

class Camera{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CAMERAID id = OTHER;
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    double target_height;
    double fx, fy, cx, cy;
    std::vector<double> k1k2Distortion;
    std::vector<double> p1p2p3Distortion;
    cv::Mat matrix = cv::Mat(3,3,CV_64F);
    cv::Mat coeff = cv::Mat(1,4,CV_64F);
    Eigen::Matrix4d Tw_c;

    Camera(const CAMERAID i):id(i){
        switch(id){
            case USBFHD01M: { // 1280x720
                fx = 1042.3;
                fy = 1042.3;
                cx = 593.2342;
                cy = 388.0768;
                k1k2Distortion[0] = -0.4403;
                k1k2Distortion[1] = 0.2877;
                p1p2p3Distortion[0] = -0.0013;
                p1p2p3Distortion[1] = -0.00074514;
                p1p2p3Distortion[2] = 0.00000;
                break;
            }
            case DAHUA1: { // 1280x1024
                fx = 1678.38;
                fy = 1675.57;
                cx = 686.14;
                cy = 445.58;
                k1k2Distortion[0] = -0.0811;
                k1k2Distortion[1] = 0.0396;
                p1p2p3Distortion[0] = -0.0116;
                p1p2p3Distortion[1] = 0.0027;
                p1p2p3Distortion[2] = 1.1588;
                cv::Mat matrix = cv::Mat(3, 3, CV_64F);
                cv::Mat coeff = cv::Mat(1, 4, CV_64F);
                break;
            }
            case DAHUA2: { // 1280x720
                fx = 1666.339;
                fy = 1675.57;
                cx = 651.1;
                cy = 490.956;
                k1k2Distortion.push_back(-0.1066);
                k1k2Distortion.push_back(0.2639);
                p1p2p3Distortion.push_back(-0.0037);
                p1p2p3Distortion.push_back(-0.0047);
                p1p2p3Distortion.push_back(0.5340);
                cv::Mat matrix = cv::Mat(3, 3, CV_64F);
                cv::Mat coeff = cv::Mat(1, 4, CV_64F);
                break;
            }
            case OTHER:{
                break;
            }
            default: {
                std::cout << "Uncorrect camera initialized !" << std::endl;
                break;
            }
        }
        paramtersInit();

    }
    // Camera(std::string filename);
    ~Camera() = default;

    void paramtersInit();
    cv::Point2d back_project(cv::Point2d &p);
    cv::Point3d undistort(cv::Point2d);
};
}

#endif //ARMOR_DETECTOR_CAMERA_H
